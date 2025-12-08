"""
Contrastive Learning Training with Dependency Triples
Stage 2: Fine-tune BERT token embeddings using syntactic structure with InfoNCE loss
Looks up embeddings dynamically from BERT so gradients can update the embedding layer
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import wandb
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# Disable tokenizer parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class RelationAwareBatchSampler(Sampler):
    """
    Custom sampler that creates batches with high probability of positive pairs.
    Groups samples by relation to ensure each batch has multiple triples from the same relation.
    """
    
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group sample indices by relation
        self.relation_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            rel = sample['relation']
            if rel not in self.relation_to_indices:
                self.relation_to_indices[rel] = []
            self.relation_to_indices[rel].append(idx)
        
        # Count true samples per relation
        self.relation_true_counts = {}
        for rel, indices in self.relation_to_indices.items():
            true_count = sum(1 for idx in indices if dataset.samples[idx]['is_true_sample'])
            self.relation_true_counts[rel] = true_count
        
        print(f"\nRelation-aware sampler initialized:")
        print(f"  Relations with 2+ true samples: {sum(1 for c in self.relation_true_counts.values() if c >= 2)}/{len(self.relation_true_counts)}")
        print(f"  Relations with 5+ true samples: {sum(1 for c in self.relation_true_counts.values() if c >= 5)}/{len(self.relation_true_counts)}")
    
    def __iter__(self):
        # Separate relations by true sample count
        rich_relations = [rel for rel, count in self.relation_true_counts.items() if count >= 4]
        medium_relations = [rel for rel, count in self.relation_true_counts.items() if 2 <= count < 4]
        poor_relations = [rel for rel, count in self.relation_true_counts.items() if count < 2]
        
        batches = []
        
        # Strategy 1: For relations with many true samples, create relation-focused batches
        for rel in rich_relations:
            indices = self.relation_to_indices[rel].copy()
            random.shuffle(indices)
            
            # Create batches from this relation
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Strategy 2: For medium relations, mix 2-3 relations per batch
        random.shuffle(medium_relations)
        combined_indices = []
        for rel in medium_relations:
            combined_indices.extend(self.relation_to_indices[rel])
            
            # When we have enough for a batch, create one
            while len(combined_indices) >= self.batch_size:
                batch = combined_indices[:self.batch_size]
                combined_indices = combined_indices[self.batch_size:]
                random.shuffle(batch)
                batches.append(batch)
        
        # Strategy 3: Mix in poor relations with random sampling
        poor_indices = []
        for rel in poor_relations:
            poor_indices.extend(self.relation_to_indices[rel])
        
        if poor_indices:
            random.shuffle(poor_indices)
            for i in range(0, len(poor_indices), self.batch_size):
                batch = poor_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Add any remaining indices
        if combined_indices and not self.drop_last:
            batches.append(combined_indices)
        
        # Shuffle batch order
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DependencyTripleDataset(Dataset):
    """Dataset for dependency triples - stores token strings, not embeddings"""
    
    def __init__(self, data_file):
        """
        Args:
            data_file: JSONL file with triples (original format)
        """
        self.samples = []
        
        print(f"Loading dataset from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sentence_data = json.loads(line.strip())
                
                for sent_id, triple_lists in sentence_data.items():
                    positive_triples, negative_triples = triple_lists
                    
                    # Add positive (true) triples
                    for triple_dict in positive_triples:
                        self.samples.append({
                            'head_token': triple_dict['triple'][0],
                            'dep_token': triple_dict['triple'][1],
                            'relation': triple_dict['triple'][2],
                            'is_true_sample': True,
                            'sentence_id': int(sent_id)
                        })
                    
                    # Add negative (artificial) triples
                    for triple_dict in negative_triples:
                        self.samples.append({
                            'head_token': triple_dict['triple'][0],
                            'dep_token': triple_dict['triple'][1],
                            'relation': triple_dict['triple'][2],
                            'is_true_sample': False,
                            'sentence_id': int(sent_id)
                        })
        
        print(f"Loaded {len(self.samples)} triples")
        
        # Statistics
        num_true = sum(1 for s in self.samples if s['is_true_sample'])
        print(f"  True samples: {num_true}")
        print(f"  Negative samples: {len(self.samples) - num_true}")
        
        # Relation statistics
        self.relations = sorted(set(s['relation'] for s in self.samples))
        relation_counts = {}
        for rel in self.relations:
            count = sum(1 for s in self.samples if s['relation'] == rel and s['is_true_sample'])
            relation_counts[rel] = count
        
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


class ContrastiveDependencyModel(nn.Module):
    """Model that looks up embeddings from BERT and trains them via backprop"""
    
    def __init__(self, bert_model, tokenizer, projection_dim=256):
        """
        Args:
            bert_model: Pretrained BERT model (embeddings will be fine-tuned)
            tokenizer: BERT tokenizer
            projection_dim: Dimension for projection head
        """
        super().__init__()
        
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        
        hidden_dim = bert_model.config.hidden_size
        triple_dim = hidden_dim * 2  # head + dep
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(triple_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
        )
        
        print(f"Model initialized:")
        print(f"  BERT hidden dim: {hidden_dim}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  BERT embeddings: trainable")
    
    def get_token_embedding(self, token_text, device):
        """
        Get embedding for a token from BERT's embedding layer.
        Gradients will flow through this to update the embeddings.
        
        Args:
            token_text: String token
            device: torch device
            
        Returns:
            embedding: [hidden_dim] tensor with gradients enabled
        """
        # Tokenize
        token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
        
        if len(token_ids) == 0:
            token_ids = [self.tokenizer.unk_token_id]
        
        token_ids_tensor = torch.tensor(token_ids, device=device)
        
        # Get embeddings from BERT's embedding layer (gradients enabled!)
        embeddings = self.bert_model.embeddings.word_embeddings(token_ids_tensor)
        
        # Average over subwords if needed
        token_embedding = embeddings.mean(dim=0)
        
        return token_embedding
    
    def forward(self, head_tokens, dep_tokens):
        """
        Forward pass - looks up embeddings dynamically
        
        Args:
            head_tokens: List of head token strings [batch_size]
            dep_tokens: List of dependent token strings [batch_size]
            
        Returns:
            projected: [batch_size, projection_dim] normalized embeddings
        """
        device = next(self.bert_model.parameters()).device
        
        # Look up embeddings from BERT (with gradients!)
        head_embeddings = []
        dep_embeddings = []
        
        for head_token, dep_token in zip(head_tokens, dep_tokens):
            head_emb = self.get_token_embedding(head_token, device)
            dep_emb = self.get_token_embedding(dep_token, device)
            
            head_embeddings.append(head_emb)
            dep_embeddings.append(dep_emb)
        
        # Stack into batch tensors
        head_embeddings = torch.stack(head_embeddings)  # [batch_size, hidden_dim]
        dep_embeddings = torch.stack(dep_embeddings)    # [batch_size, hidden_dim]
        
        # Concatenate head and dependent embeddings
        triple_repr = torch.cat([head_embeddings, dep_embeddings], dim=1)
        
        # Project to contrastive space
        projected = self.projection(triple_repr)
        
        # L2 normalize for cosine similarity
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for dependency triples"""
    
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, relations, is_true_samples):
        """
        Compute InfoNCE loss
        
        Args:
            embeddings: [batch_size, projection_dim] - normalized embeddings
            relations: List of relation strings [batch_size]
            is_true_samples: [batch_size] - True for real triples, False for negative samples
            
        Returns:
            loss: Scalar loss value
            num_positives: Number of samples with positive pairs
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Create positive pairs mask
        # Positive pairs: same relation AND both are true samples
        positive_pairs = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            if is_true_samples[i]:
                for j in range(batch_size):
                    if i != j and is_true_samples[j] and relations[i] == relations[j]:
                        positive_pairs[i, j] = True
        
        # Count samples with positive pairs
        num_positives = positive_pairs.any(dim=1).sum().item()
        
        # If no positive pairs in batch, return zero loss
        if not positive_pairs.any():
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # Compute loss for samples that have positive pairs
        losses = []
        for i in range(batch_size):
            if positive_pairs[i].any():
                pos_sim = similarity_matrix[i][positive_pairs[i]]
                numerator = torch.exp(pos_sim).sum()
                denominator = sum_exp_sim[i]
                loss_i = -torch.log(numerator / (denominator + 1e-8))
                losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        return torch.stack(losses).mean(), num_positives


def collate_fn(batch):
    """Custom collate function to handle string tokens"""
    return {
        'head_token': [item['head_token'] for item in batch],
        'dep_token': [item['dep_token'] for item in batch],
        'relation': [item['relation'] for item in batch],
        'is_true_sample': torch.tensor([item['is_true_sample'] for item in batch]),
    }


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches_with_loss = 0
    total_positives = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move tensors to device
        is_true_samples = batch['is_true_sample'].to(device)
        
        # Forward pass - this looks up embeddings from BERT dynamically
        projected = model(batch['head_token'], batch['dep_token'])
        
        # Compute contrastive loss
        loss, num_pos = criterion(projected, batch['relation'], is_true_samples)
        
        # Skip if no positive pairs in batch
        if loss.item() == 0.0:
            continue
        
        # Backward pass - gradients flow back to BERT embeddings!
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches_with_loss += 1
        total_positives += num_pos
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'pos_pairs': num_pos
        })
        
        # Log to wandb
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
    
    parser = argparse.ArgumentParser(description='Contrastive learning with dependency triples')
    parser.add_argument('--data_file', type=str, required=True,
                        help='JSONL file with triples (original format)')
    parser.add_argument('--bert_model', type=str, required=True,
                        help='Path to BERT model from Stage 1')
    parser.add_argument('--output_dir', type=str, default='./contrastive_model',
                        help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature for InfoNCE loss')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Projection head dimension')
    parser.add_argument('--warmup_steps', type=int, default=200,
                        help='Number of warmup steps')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="bert-contrastive-dependency",
        config=vars(args)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BERT model and tokenizer
    print("Loading BERT model...")
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    # Create dataset and dataloader
    dataset = DependencyTripleDataset(args.data_file)
    
    # Use relation-aware batch sampler
    batch_sampler = RelationAwareBatchSampler(
        dataset,
        batch_size=args.batch_size,
        drop_last=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,  # Use custom sampler instead of shuffle
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = ContrastiveDependencyModel(
        bert_model=bert_model,
        tokenizer=tokenizer,
        projection_dim=args.projection_dim
    ).to(device)
    
    # Optimizer - trains BOTH the projection head AND BERT embeddings
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss
    criterion = InfoNCELoss(temperature=args.temperature)
    
    # Training loop
    print("\nStarting training...")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    for epoch in range(1, args.epochs + 1):
        avg_loss, num_batches, avg_pos = train_epoch(
            model, dataloader, optimizer, scheduler, criterion, device, epoch
        )
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}, "
              f"Batches with loss: {num_batches}/{len(dataloader)}, "
              f"Avg positive pairs: {avg_pos:.1f}")
        
        wandb.log({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'batches_with_loss': num_batches,
            'avg_positive_pairs': avg_pos
        })
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save BERT model (the embeddings we care about)
            bert_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save full training state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': vars(args)
            }, os.path.join(checkpoint_path, 'training_state.pt'))
            
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    
    # Save BERT model and tokenizer (for downstream use)
    bert_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save full training state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args)
    }, os.path.join(final_path, 'training_state.pt'))
    
    print(f"\n✅ Training complete! Final BERT model saved to: {final_path}")
    print(f"Use this model for downstream NLP tasks in Tagalog")
    
    wandb.finish()


if __name__ == "__main__":
    main()
