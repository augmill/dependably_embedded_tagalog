import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QWEN_MODELS = {
    "qwen_0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen_7b": "Qwen/Qwen2.5-7B",
}

MAX_NEW_TOKENS = 32


def load_fake_news_data():
    """Load and split fake news data (same splits as classifier/run.py)."""
    fake_news_data = load_dataset("jcblaise/fake_news_filipino")["train"]
    data = [(item["article"], item["label"]) for item in fake_news_data]
    X_train, X_test, y_train, y_test = train_test_split(
        [d[0] for d in data],
        [d[1] for d in data],
        train_size=0.8,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    split = int(len(X_test) * 0.5)
    X_val, y_val = X_test[:split], y_test[:split]
    X_test, y_test = X_test[split + 1 :], y_test[split + 1 :]
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_few_shot_examples(X_train, y_train, n=5):
    """Select balanced few-shot examples from training data."""
    real_examples = [(x, y) for x, y in zip(X_train, y_train) if y == 0]
    fake_examples = [(x, y) for x, y in zip(X_train, y_train) if y == 1]

    # Take roughly equal from each class
    n_per_class = n // 2
    examples = real_examples[:n_per_class] + fake_examples[: n - n_per_class]
    return examples


def build_prompt(article, few_shot_examples=None):
    instruction = (
        "Classify the following Filipino news article as either \"real\" or \"fake\". "
        "Respond with ONLY the word \"real\" or \"fake\"."
    )

    prompt = f"Instruction: {instruction}\n\n"

    if few_shot_examples:
        for text, label in few_shot_examples:
            label_str = "real" if label == 0 else "fake"
            # Truncate examples to keep prompt manageable
            truncated = text[:500] + "..." if len(text) > 500 else text
            prompt += f"Article: {truncated}\nClassification: {label_str}\n\n"

    # Truncate test article too
    truncated_article = article[:1000] + "..." if len(article) > 1000 else article
    prompt += f"Article: {truncated_article}\nClassification:"
    return prompt


def parse_classification(response):
    """Parse model response into binary label."""
    response_lower = response.strip().lower()
    if "fake" in response_lower:
        return 1
    elif "real" in response_lower:
        return 0
    # Default to most common class if unparseable
    return -1


def evaluate_qwen_classification(model_key, model_path, few_shot_examples=None):
    print("=" * 60)
    mode = "few-shot" if few_shot_examples else "zero-shot"
    print(f"Fake News Classification: {model_key} ({mode})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, _, _, _, X_test, y_test = load_fake_news_data()

    all_preds = []
    all_labels = []
    unparseable = 0

    for article, label in tqdm(zip(X_test, y_test), total=len(X_test), desc=f"Evaluating {model_key}"):
        prompt = build_prompt(article, few_shot_examples=few_shot_examples)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        pred = parse_classification(response)
        if pred == -1:
            unparseable += 1
            pred = 0  # Default fallback
        all_preds.append(pred)
        all_labels.append(label)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    report = classification_report(all_labels, all_preds, target_names=["real", "fake"])

    print(f"\nResults for {model_key} ({mode}):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Unparseable responses: {unparseable}/{len(all_preds)}")
    print(f"\n{report}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"accuracy": acc, "f1": f1, "unparseable": unparseable, "report": report}


if __name__ == "__main__":
    X_train, y_train, _, _, _, _ = load_fake_news_data()
    few_shot_examples = get_few_shot_examples(X_train, y_train, n=5)

    results = {}

    for model_key, model_path in QWEN_MODELS.items():
        for mode, examples in [("zero_shot", None), ("few_shot", few_shot_examples)]:
            key = f"{model_key}_{mode}"
            try:
                results[key] = evaluate_qwen_classification(model_key, model_path, few_shot_examples=examples)
            except Exception as e:
                print(f"Error running {key}: {e}")
                results[key] = None

    print("\n" + "=" * 60)
    print("SUMMARY: Qwen Fake News Classification Results")
    print("=" * 60)
    for key, res in results.items():
        if res:
            print(f"  {key:30s}: Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
        else:
            print(f"  {key:30s}: FAILED")
