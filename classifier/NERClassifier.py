import torch
import torch.nn as nn
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

seed = 42
torch.manual_seed(seed)

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(NER_LABELS)}
NUM_LABELS = len(NER_LABELS)
IGNORE_INDEX = -100


class NERClassifier(nn.Module):
    def __init__(self, dim_in: int = 768, num_labels: int = NUM_LABELS, drop: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(dim_in, num_labels),
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.classifier(hidden_states)


def ner_training(model, data, opt, loss_fn, device):
    losses = []
    model.train()
    for batch in tqdm(data, desc="Training"):
        hidden_states, labels = batch[0].to(device), batch[1].to(device)
        opt.zero_grad()
        logits = model(hidden_states)
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)
        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    return losses


def ner_evaluate(model, data, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating"):
            hidden_states, labels = batch[0].to(device), batch[1].to(device)
            logits = model(hidden_states)
            preds = torch.argmax(logits, dim=-1)

            for i in range(labels.shape[0]):
                pred_seq = []
                label_seq = []
                for j in range(labels.shape[1]):
                    if labels[i, j].item() != IGNORE_INDEX:
                        pred_seq.append(ID2LABEL[preds[i, j].item()])
                        label_seq.append(ID2LABEL[labels[i, j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    overall_f1 = f1_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds)
    overall_recall = recall_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return {
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
        "report": report,
    }


def fit_ner(model, train_data, dev_data, opt, loss_fn, epochs, device):
    best_f1 = 0.0
    best_state = None
    for epoch in range(epochs):
        print("-" * 25 + f" epoch {epoch + 1} " + "-" * 25)
        train_losses = ner_training(model, train_data, opt, loss_fn, device)
        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Training loss: {avg_loss:.4f}")

        results = ner_evaluate(model, dev_data, device)
        print(f"Dev F1: {results['f1']:.4f}  Precision: {results['precision']:.4f}  Recall: {results['recall']:.4f}")

        if results["f1"] > best_f1:
            best_f1 = results["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
