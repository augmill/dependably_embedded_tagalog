import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from NERClassifier import (
    NERClassifier, fit_ner, ner_evaluate,
    LABEL2ID, NUM_LABELS, IGNORE_INDEX,
)

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
BERT_MODELS = {
    "base": "google-bert/bert-base-multilingual-cased",
    "mlm_only": "paulbontempo/bert-tagalog-mlm-stage1",
    "cl_only": "paulbontempo/bert-tagalog-cl-only",       # Update after uploading
    "mlm_cl": "paulbontempo/bert-tagalog-dependency-cl",
}

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DROPOUT = 0.3
MAX_LEN = 256


# --- Data loading & tokenization ---
def load_ner_dataset():
    ds = load_dataset("ljvmiranda921/tlunified-ner")
    return ds["train"], ds["validation"], ds["test"]


def align_labels_with_tokens(labels, word_ids):
    """Map word-level NER labels to subword tokens (IOB2-aware)."""
    aligned = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            aligned.append(IGNORE_INDEX)
        elif wid != prev_word_id:
            aligned.append(labels[wid])
        else:
            # For continuation subwords: use I- tag if the word label is B-
            label = labels[wid]
            if label % 2 == 1:  # B- tags are odd indices (B-PER=1, B-ORG=3, B-LOC=5)
                aligned.append(label + 1)  # Convert B- to I-
            else:
                aligned.append(label)
        prev_word_id = wid
    return aligned


def extract_embeddings(dataset_split, tokenizer, bert_model, batch_size, max_len):
    """Extract BERT last_hidden_state embeddings and aligned NER labels."""
    bert_model.eval()
    all_hidden = []
    all_labels = []

    tokens_list = dataset_split["tokens"]
    ner_tags_list = dataset_split["ner_tags"]

    for i in range(0, len(tokens_list), batch_size):
        batch_tokens = tokens_list[i : i + batch_size]
        batch_tags = ner_tags_list[i : i + batch_size]

        encoding = tokenizer(
            batch_tokens,
            is_pretokenized=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

        batch_aligned_labels = []
        for j, tags in enumerate(batch_tags):
            word_ids = encoding.word_ids(batch_index=j)
            aligned = align_labels_with_tokens(tags, word_ids)
            # Pad/truncate to max_len
            aligned = aligned[:max_len]
            aligned += [IGNORE_INDEX] * (max_len - len(aligned))
            batch_aligned_labels.append(aligned)

        with torch.no_grad():
            outputs = bert_model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
            hidden_states = outputs.last_hidden_state.cpu()

        labels_tensor = torch.tensor(batch_aligned_labels, dtype=torch.long)
        all_hidden.append(hidden_states)
        all_labels.append(labels_tensor)

    return list(zip(
        torch.cat(all_hidden).split(batch_size),
        torch.cat(all_labels).split(batch_size),
    ))


# --- Main ---
def run_ner_for_model(model_key, model_path):
    print("=" * 60)
    print(f"NER Evaluation: {model_key} ({model_path})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bert = AutoModel.from_pretrained(model_path).to(device)
    for param in bert.parameters():
        param.requires_grad = False

    train_split, dev_split, test_split = load_ner_dataset()

    print("Extracting train embeddings...")
    train_batches = extract_embeddings(train_split, tokenizer, bert, BATCH_SIZE, MAX_LEN)
    print("Extracting dev embeddings...")
    dev_batches = extract_embeddings(dev_split, tokenizer, bert, BATCH_SIZE, MAX_LEN)
    print("Extracting test embeddings...")
    test_batches = extract_embeddings(test_split, tokenizer, bert, BATCH_SIZE, MAX_LEN)

    # Free BERT from GPU
    del bert
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ner_model = NERClassifier(dim_in=768, num_labels=NUM_LABELS, drop=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(ner_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print("\nTraining NER head...")
    ner_model = fit_ner(ner_model, train_batches, dev_batches, optimizer, loss_fn, EPOCHS, device)

    print("\nTest set evaluation:")
    test_results = ner_evaluate(ner_model, test_batches, device)
    print(f"Test F1: {test_results['f1']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print("\nDetailed report:")
    print(test_results["report"])

    return test_results


if __name__ == "__main__":
    results = {}
    for model_key, model_path in BERT_MODELS.items():
        try:
            results[model_key] = run_ner_for_model(model_key, model_path)
        except Exception as e:
            print(f"Error running {model_key}: {e}")
            results[model_key] = None

    print("\n" + "=" * 60)
    print("SUMMARY: NER Results (span-level F1)")
    print("=" * 60)
    for model_key, res in results.items():
        if res:
            print(f"  {model_key:12s}: F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}")
        else:
            print(f"  {model_key:12s}: FAILED")
