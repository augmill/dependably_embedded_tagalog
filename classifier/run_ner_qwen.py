import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QWEN_MODELS = {
    "qwen_0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen_7b": "Qwen/Qwen2.5-7B",
}

MAX_NEW_TOKENS = 512

FEW_SHOT_EXAMPLES = [
    {
        "text": "Si Juan dela Cruz ay nagtrabaho sa Department of Education sa Maynila.",
        "entities": {"PER": ["Juan dela Cruz"], "ORG": ["Department of Education"], "LOC": ["Maynila"]},
    },
    {
        "text": "Ang Philippine Red Cross ay nagpadala ng tulong sa Tacloban.",
        "entities": {"PER": [], "ORG": ["Philippine Red Cross"], "LOC": ["Tacloban"]},
    },
    {
        "text": "Walang nakitang anumang espesyal na bagay sa lugar.",
        "entities": {"PER": [], "ORG": [], "LOC": []},
    },
]


def build_prompt(text, few_shot=True):
    instruction = (
        "Extract named entities from the following Tagalog text. "
        "Identify all PER (person), ORG (organization), and LOC (location) entities. "
        "Return ONLY a JSON object with keys \"PER\", \"ORG\", \"LOC\", each mapping to a list of entity strings. "
        "If no entities of a type are found, use an empty list."
    )

    prompt = f"Instruction: {instruction}\n\n"

    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            prompt += f"Text: {ex['text']}\n"
            prompt += f"Entities: {json.dumps(ex['entities'])}\n\n"

    prompt += f"Text: {text}\nEntities:"
    return prompt


def parse_entities(output_text):
    """Parse JSON entity output from model response."""
    # Try to find JSON in the output
    json_match = re.search(r'\{[^{}]*\}', output_text)
    if json_match:
        try:
            entities = json.loads(json_match.group())
            result = {}
            for key in ["PER", "ORG", "LOC"]:
                val = entities.get(key, [])
                if isinstance(val, list):
                    result[key] = [str(v) for v in val]
                else:
                    result[key] = []
            return result
        except json.JSONDecodeError:
            pass
    return {"PER": [], "ORG": [], "LOC": []}


def tokens_to_text(tokens):
    return " ".join(tokens)


def get_gold_entities(tokens, ner_tags, id2label):
    """Convert IOB2 tags to entity spans."""
    entities = {"PER": [], "ORG": [], "LOC": []}
    current_entity = []
    current_type = None

    for token, tag_id in zip(tokens, ner_tags):
        tag = id2label[tag_id]
        if tag.startswith("B-"):
            if current_entity and current_type:
                entities[current_type].append(" ".join(current_entity))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(token)
        else:
            if current_entity and current_type:
                entities[current_type].append(" ".join(current_entity))
            current_entity = []
            current_type = None

    if current_entity and current_type:
        entities[current_type].append(" ".join(current_entity))

    return entities


def entities_to_iob2(tokens, entities):
    """Convert extracted entity strings back to IOB2 tags for seqeval comparison."""
    tags = ["O"] * len(tokens)
    text = " ".join(tokens)

    for ent_type in ["PER", "ORG", "LOC"]:
        for entity in entities.get(ent_type, []):
            ent_tokens = entity.split()
            for i in range(len(tokens) - len(ent_tokens) + 1):
                if tokens[i : i + len(ent_tokens)] == ent_tokens:
                    tags[i] = f"B-{ent_type}"
                    for j in range(1, len(ent_tokens)):
                        tags[i + j] = f"I-{ent_type}"
                    break  # First match only

    return tags


def evaluate_qwen_ner(model_key, model_path, few_shot=True, max_samples=None):
    print("=" * 60)
    print(f"NER Evaluation: {model_key} ({model_path})")
    print(f"Mode: {'few-shot' if few_shot else 'zero-shot'}")
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

    ds = load_dataset("ljvmiranda921/tlunified-ner")
    test_data = ds["test"]

    # Label mapping from dataset
    label_names = test_data.features["ner_tags"].feature.names
    id2label = {i: name for i, name in enumerate(label_names)}

    all_gold_tags = []
    all_pred_tags = []

    samples = list(range(len(test_data)))
    if max_samples:
        samples = samples[:max_samples]

    for idx in tqdm(samples, desc=f"Evaluating {model_key}"):
        tokens = test_data[idx]["tokens"]
        ner_tags = test_data[idx]["ner_tags"]

        text = tokens_to_text(tokens)
        prompt = build_prompt(text, few_shot=few_shot)

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

        pred_entities = parse_entities(response)
        pred_tags = entities_to_iob2(tokens, pred_entities)

        gold_tags = [id2label[t] for t in ner_tags]

        all_gold_tags.append(gold_tags)
        all_pred_tags.append(pred_tags)

    overall_f1 = f1_score(all_gold_tags, all_pred_tags)
    overall_p = precision_score(all_gold_tags, all_pred_tags)
    overall_r = recall_score(all_gold_tags, all_pred_tags)
    report = classification_report(all_gold_tags, all_pred_tags)

    print(f"\nResults for {model_key}:")
    print(f"  F1: {overall_f1:.4f}  Precision: {overall_p:.4f}  Recall: {overall_r:.4f}")
    print("\nDetailed report:")
    print(report)

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"f1": overall_f1, "precision": overall_p, "recall": overall_r, "report": report}


if __name__ == "__main__":
    results = {}

    for model_key, model_path in QWEN_MODELS.items():
        for mode, few_shot in [("few_shot", True), ("zero_shot", False)]:
            key = f"{model_key}_{mode}"
            try:
                results[key] = evaluate_qwen_ner(model_key, model_path, few_shot=few_shot)
            except Exception as e:
                print(f"Error running {key}: {e}")
                results[key] = None

    print("\n" + "=" * 60)
    print("SUMMARY: Qwen NER Results (span-level F1)")
    print("=" * 60)
    for key, res in results.items():
        if res:
            print(f"  {key:30s}: F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}")
        else:
            print(f"  {key:30s}: FAILED")
