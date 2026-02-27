"""
Batayan Abstractive Summarization Evaluation
Evaluates:
  - Qwen-0.5B baseline (zero/few-shot, no CL fine-tuning)
  - Qwen-0.5B+CL (embed_tokens fine-tuned with Tagalog dependency triples)
  - Qwen-1.5B and Qwen-7B zero/few-shot (reference points)

Dataset: Batayan Abstractive Summarization (part of SEA-HELM)
Metrics: BERTScore, ChrF++, ROUGE-L F1
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from bert_score import score as bert_score
from sacrebleu.metrics import CHRF
from rouge_score import rouge_scorer
from tqdm import tqdm

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model configs ---
# HuggingFace repo for the CL-fine-tuned Qwen-0.5B (update after upload)
QWEN_CL_MODEL = "paulbontempo/qwen-0.5b-tagalog-cl"

MODELS = {
    "qwen_0.5b_baseline": "Qwen/Qwen2.5-0.5B",
    "qwen_0.5b_cl": QWEN_CL_MODEL,
    "qwen_1.5b_baseline": "Qwen/Qwen2.5-1.5B",
    "qwen_7b_baseline": "Qwen/Qwen2.5-7B",
}

MAX_NEW_TOKENS = 256

FEW_SHOT_EXAMPLES = [
    {
        "article": "Inanunsyo ng Pangulo ang bagong programa para sa mga magsasaka sa Mindanao. "
                   "Kasama sa programa ang libreng binhi at pataba para sa mga benepisyaryo. "
                   "Tinatayang aabot sa sampung libong pamilya ang makikinabang sa inisyatibang ito.",
        "summary": "Naglunsad ang Pangulo ng programa para sa mga magsasaka sa Mindanao na "
                   "kinabibilangan ng libreng binhi at pataba para sa sampung libong pamilya.",
    },
    {
        "article": "Nagtala ang Bangko Sentral ng Pilipinas ng pagtaas ng inflation rate noong nakaraang buwan. "
                   "Ayon sa ulat, umabot ito sa 4.2 porsyento mula sa 3.8 porsyento noong nakaraang buwan. "
                   "Pangunahing dahilan nito ang pagtaas ng presyo ng pagkain at gasolina.",
        "summary": "Tumaas ang inflation rate sa 4.2 porsyento dahil sa mas mataas na presyo ng pagkain at gasolina.",
    },
]


def build_prompt(article, few_shot=True):
    instruction = (
        "Sumulat ng maikling buod ng sumusunod na balita sa Filipino. "
        "Ang buod ay dapat na isa o dalawang pangungusap lamang."
    )
    prompt = f"Instruksyon: {instruction}\n\n"

    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            prompt += f"Balita: {ex['article']}\nBuod: {ex['summary']}\n\n"

    prompt += f"Balita: {article}\nBuod:"
    return prompt


def generate_summary(model, tokenizer, article, few_shot=True, max_input_len=1024):
    prompt = build_prompt(article, few_shot=few_shot)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Take only the first line/sentence if model produces more
    summary = summary.split("\n")[0].strip()
    return summary


def compute_metrics(predictions, references):
    """Compute BERTScore, ChrF++, and ROUGE-L."""
    # BERTScore (use multilingual model)
    P, R, F1 = bert_score(
        predictions, references,
        lang="tl",  # Tagalog
        model_type="bert-base-multilingual-cased",
        verbose=False,
    )
    bertscore_f1 = F1.mean().item()

    # ChrF++
    chrf = CHRF(word_order=2)  # word_order=2 gives ChrF++
    chrf_score = chrf.corpus_score(predictions, [references]).score

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                    for pred, ref in zip(predictions, references)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    return {
        "bertscore_f1": bertscore_f1,
        "chrf_pp": chrf_score,
        "rouge_l": rouge_l,
    }


def evaluate_model(model_key, model_path, test_data, few_shot=True, max_samples=None):
    print("=" * 60)
    mode = "few-shot" if few_shot else "zero-shot"
    print(f"Evaluating: {model_key} ({mode})")
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

    articles = test_data["text"] if "text" in test_data.column_names else test_data["article"]
    references = test_data["summary"] if "summary" in test_data.column_names else test_data["highlights"]

    if max_samples:
        articles = articles[:max_samples]
        references = references[:max_samples]

    predictions = []
    for article in tqdm(articles, desc=f"Generating ({model_key})"):
        pred = generate_summary(model, tokenizer, article, few_shot=few_shot)
        predictions.append(pred)

    metrics = compute_metrics(predictions, list(references))

    print(f"\nResults for {model_key} ({mode}):")
    print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"  ChrF++:       {metrics['chrf_pp']:.2f}")
    print(f"  ROUGE-L:      {metrics['rouge_l']:.4f}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


if __name__ == "__main__":
    # Load Batayan Abstractive Summarization dataset
    # Dataset: aisingapore/sea-bench or the Batayan-specific split
    # Update the dataset path/split name once confirmed on HuggingFace
    print("Loading Batayan Abstractive Summarization dataset...")
    try:
        ds = load_dataset("aisingapore/sea-helm", "fil_abstractive_summarization")
        test_data = ds["test"]
    except Exception as e:
        print(f"Could not load dataset automatically: {e}")
        print("Please update the dataset path in this script once confirmed on HuggingFace.")
        raise

    results = {}
    for model_key, model_path in MODELS.items():
        # CL model uses zero-shot (no in-context examples needed — it has structural priors)
        # Baseline models use few-shot for fair comparison
        few_shot = "cl" not in model_key
        key = f"{model_key}_{'few_shot' if few_shot else 'zero_shot'}"
        try:
            results[key] = evaluate_model(model_key, model_path, test_data, few_shot=few_shot)
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
            results[key] = None

    print("\n" + "=" * 60)
    print("SUMMARY: Batayan Abstractive Summarization Results")
    print("=" * 60)
    print(f"{'Model':<35} {'BERTScore':>10} {'ChrF++':>8} {'ROUGE-L':>8}")
    print("-" * 65)
    for key, res in results.items():
        if res:
            print(f"  {key:<33} {res['bertscore_f1']:>10.4f} {res['chrf_pp']:>8.2f} {res['rouge_l']:>8.4f}")
        else:
            print(f"  {key:<33} {'FAILED':>10}")
