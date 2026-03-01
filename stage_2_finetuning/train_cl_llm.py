"""
Contrastive Learning Training for Decoder-Only LLMs (e.g., Qwen)

Two training modes selected via --full_backprop flag:

  embed_only (default):
    Looks up static token embeddings from model.model.embed_tokens.
    Only embed_tokens + projection head are trainable. Fast, low memory.
    Analogous to the BERT CL approach in train_cl.py.

  full_backprop (--full_backprop):
    Runs the full LLM forward pass and uses last_hidden_state as the
    token representation. All model parameters are trainable. The CL
    signal propagates through every transformer layer, teaching the model
    to produce similar *contextual* representations for tokens that share
    the same syntactic dependency relation.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class RelationAwareBatchSampler(Sampler):
    """
    Custom sampler that creates batches with high probability of positive pairs.
    Groups samples by relation to ensure each batch has multiple triples from the same relation.
    (Identical to train_cl.py)
    """

    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.relation_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            rel = sample['relation']
            if rel not in self.relation_to_indices:
                self.relation_to_indices[rel] = []
            self.relation_to_indices[rel].append(idx)

        self.relation_true_counts = {}
        for rel, indices in self.relation_to_indices.items():
            true_count = sum(1 for idx in indices if dataset.samples[idx]['is_true_sample'])
            self.relation_true_counts[rel] = true_count

        print(f"\nRelation-aware sampler initialized:")
        print(f"  Relations with 2+ true samples: {sum(1 for c in self.relation_true_counts.values() if c >= 2)}/{len(self.relation_true_counts)}")
        print(f"  Relations with 5+ true samples: {sum(1 for c in self.relation_true_counts.values() if c >= 5)}/{len(self.relation_true_counts)}")

    def __iter__(self):
        rich_relations = [rel for rel, count in self.relation_true_counts.items() if count >= 4]
        medium_relations = [rel for rel, count in self.relation_true_counts.items() if 2 <= count < 4]
        poor_relations = [rel for rel, count in self.relation_true_counts.items() if count < 2]

        batches = []

        for rel in rich_relations:
            indices = self.relation_to_indices[rel].copy()
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        random.shuffle(medium_relations)
        combined_indices = []
        for rel in medium_relations:
            combined_indices.extend(self.relation_to_indices[rel])
            while len(combined_indices) >= self.batch_size:
                batch = combined_indices[:self.batch_size]
                combined_indices = combined_indices[self.batch_size:]
                random.shuffle(batch)
                batches.append(batch)

        poor_indices = []
        for rel in poor_relations:
            poor_indices.extend(self.relation_to_indices[rel])
        if poor_indices:
            random.shuffle(poor_indices)
            for i in range(0, len(poor_indices), self.batch_size):
                batch = poor_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if combined_indices and not self.drop_last:
            batches.append(combined_indices)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DependencyTripleDataset(Dataset):
    """Dataset for dependency triples. (Identical to train_cl.py)"""

    def __init__(self, data_file):
        self.samples = []
        print(f"Loading dataset from: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sentence_data = json.loads(line.strip())
                for sent_id, triple_lists in sentence_data.items():
                    positive_triples, negative_triples = triple_lists
                    for triple_dict in positive_triples:
                        self.samples.append({
                            'head_token': triple_dict['triple'][0],
                            'dep_token': triple_dict['triple'][1],
                            'relation': triple_dict['triple'][2],
                            'is_true_sample': True,
                            'sentence_id': int(sent_id)
                        })
                    for triple_dict in negative_triples:
                        self.samples.append({
                            'head_token': triple_dict['triple'][0],
                            'dep_token': triple_dict['triple'][1],
                            'relation': triple_dict['triple'][2],
                            'is_true_sample': False,
                            'sentence_id': int(sent_id)
                        })

        print(f"Loaded {len(self.samples)} triples")
        num_true = sum(1 for s in self.samples if s['is_true_sample'])
        print(f"  True samples: {num_true}")
        print(f"  Negative samples: {len(self.samples) - num_true}")

        self.relations = sorted(set(s['relation'] for s in self.samples))
        relation_counts = {rel: sum(1 for s in self.samples if s['relation'] == rel and s['is_true_sample'])
                           for rel in self.relations}
        print(f"  Unique relations: {len(self.relations)}")
        if relation_counts:
            print(f"  True samples per relation: min={min(relation_counts.values())}, "
                  f"max={max(relation_counts.values())}, "
                  f"avg={sum(relation_counts.values())/len(relation_counts):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'head_token': sample['head_token'],
            'dep_token': sample['dep_token'],
            'relation': sample['relation'],
            'is_true_sample': sample['is_true_sample'],
        }


class ContrastiveLLMModel(nn.Module):
    """
    Contrastive model for decoder-only LLMs with two operating modes:

    embed_only (full_backprop=False):
      Looks up token vectors directly from model.model.embed_tokens —
      identical in spirit to the BERT approach in train_cl.py. Static
      (non-contextual) representations; only embed_tokens is trained.

    full_backprop (full_backprop=True):
      Runs a full causal LM forward pass on each token and extracts
      last_hidden_state. This gives *contextual* representations and
      allows gradients to update every transformer layer.
    """

    def __init__(self, llm_model, tokenizer, projection_dim=256, full_backprop=False):
        super().__init__()
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.full_backprop = full_backprop

        hidden_dim = llm_model.config.hidden_size
        triple_dim = hidden_dim * 2  # head + dep concatenated

        self.projection = nn.Sequential(
            nn.Linear(triple_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
        )

        mode_str = "full forward pass (contextual)" if full_backprop else "embed_tokens lookup (static)"
        print(f"Model initialized:")
        print(f"  LLM hidden dim: {hidden_dim}")
        print(f"  Projection input dim: {triple_dim}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Mode: {mode_str}")

    def get_token_embedding(self, token_text, device):
        """
        Get a representation for a single token string.

        embed_only: direct lookup from embed_tokens (no transformer layers).
        full_backprop: full causal LM forward pass; mean-pool last_hidden_state
          over subword positions to get a single vector per token.
        """
        token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = [self.tokenizer.unk_token_id]

        token_ids_tensor = torch.tensor(token_ids, device=device)

        if self.full_backprop:
            # Full forward pass through all transformer layers.
            # input: [1, seq_len] -> hidden_states[-1]: [1, seq_len, hidden_dim]
            outputs = self.llm_model(
                token_ids_tensor.unsqueeze(0),
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            return last_hidden.mean(dim=0).float()      # cast to float32 to match projection head
        else:
            # Static embedding lookup only
            embeddings = self.llm_model.model.embed_tokens(token_ids_tensor)
            return embeddings.mean(dim=0)

    def forward(self, head_tokens, dep_tokens):
        device = next(self.llm_model.parameters()).device

        head_embeddings = [self.get_token_embedding(t, device) for t in head_tokens]
        dep_embeddings = [self.get_token_embedding(t, device) for t in dep_tokens]

        head_embeddings = torch.stack(head_embeddings)
        dep_embeddings = torch.stack(dep_embeddings)

        triple_repr = torch.cat([head_embeddings, dep_embeddings], dim=1)
        projected = self.projection(triple_repr)
        return F.normalize(projected, p=2, dim=1)


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss. (Identical to train_cl.py)"""

    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, relations, is_true_samples):
        batch_size = embeddings.shape[0]
        device = embeddings.device

        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        mask = torch.eye(batch_size, device=device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        positive_pairs = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            if is_true_samples[i]:
                for j in range(batch_size):
                    if i != j and is_true_samples[j] and relations[i] == relations[j]:
                        positive_pairs[i, j] = True

        num_positives = positive_pairs.any(dim=1).sum().item()
        if not positive_pairs.any():
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)

        losses = []
        for i in range(batch_size):
            if positive_pairs[i].any():
                pos_sim = similarity_matrix[i][positive_pairs[i]]
                numerator = torch.exp(pos_sim).sum()
                losses.append(-torch.log(numerator / (sum_exp_sim[i] + 1e-8)))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        return torch.stack(losses).mean(), num_positives


def collate_fn(batch):
    return {
        'head_token': [item['head_token'] for item in batch],
        'dep_token': [item['dep_token'] for item in batch],
        'relation': [item['relation'] for item in batch],
        'is_true_sample': torch.tensor([item['is_true_sample'] for item in batch]),
    }


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0
    num_batches_with_loss = 0
    total_positives = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        is_true_samples = batch['is_true_sample'].to(device)
        projected = model(batch['head_token'], batch['dep_token'])
        loss, num_pos = criterion(projected, batch['relation'], is_true_samples)

        if loss.item() == 0.0:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches_with_loss += 1
        total_positives += num_pos

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'pos_pairs': num_pos})

        if num_batches_with_loss % 10 == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'positive_pairs': num_pos,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    avg_loss = total_loss / max(num_batches_with_loss, 1)
    avg_positives = total_positives / max(num_batches_with_loss, 1)
    return avg_loss, num_batches_with_loss, avg_positives


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Contrastive learning for decoder-only LLMs')
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--llm_model', type=str, required=True,
                        help='HuggingFace model ID (e.g., Qwen/Qwen2.5-0.5B)')
    parser.add_argument('--output_dir', type=str, default='./cl_llm_model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=300)
    parser.add_argument('--full_backprop', action='store_true',
                        help='Run full LLM forward pass and train all parameters. '
                             'Default: only embed_tokens is trained (static embeddings).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(project="llm-contrastive-dependency", config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training mode: {'full_backprop (contextual)' if args.full_backprop else 'embed_only (static)'}")

    print("Loading LLM...")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.float16 if args.full_backprop else torch.float32,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.full_backprop:
        # All parameters trainable; use gradient checkpointing to manage memory
        for param in llm.parameters():
            param.requires_grad = True
        llm.gradient_checkpointing_enable()
    else:
        # Only the static input embedding table is trainable
        for param in llm.parameters():
            param.requires_grad = False
        for param in llm.model.embed_tokens.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in llm.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = DependencyTripleDataset(args.data_file)
    batch_sampler = RelationAwareBatchSampler(dataset, batch_size=args.batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_fn)

    model = ContrastiveLLMModel(
        llm_model=llm,
        tokenizer=tokenizer,
        projection_dim=args.projection_dim,
        full_backprop=args.full_backprop,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    criterion = InfoNCELoss(temperature=args.temperature)

    print(f"\nStarting training...")
    print(f"Total steps: {total_steps}, Warmup: {args.warmup_steps}, Batches/epoch: {len(dataloader)}")

    for epoch in range(1, args.epochs + 1):
        avg_loss, num_batches, avg_pos = train_epoch(
            model, dataloader, optimizer, scheduler, criterion, device, epoch
        )
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}, "
              f"Batches with loss: {num_batches}/{len(dataloader)}, "
              f"Avg positive pairs: {avg_pos:.1f}")
        wandb.log({'epoch': epoch, 'avg_loss': avg_loss,
                   'batches_with_loss': num_batches, 'avg_positive_pairs': avg_pos})

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            llm.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': vars(args)
            }, os.path.join(checkpoint_path, 'training_state.pt'))
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    llm.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    torch.save({'model_state_dict': model.state_dict(), 'config': vars(args)},
               os.path.join(final_path, 'training_state.pt'))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
