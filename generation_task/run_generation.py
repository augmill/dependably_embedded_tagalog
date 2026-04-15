"""
BalitaNLP Generation Task Evaluation
Evaluates Qwen models on two tasks using the BalitaNLP Filipino news dataset
(LanceBunag/BalitaNLP, config: no-image).

Task 1 - Perplexity (primary):
  Compute mean per-token NLL over held-out article bodies.
  Lower perplexity = better LM fit to Filipino text.
  Fast: no generation needed.

Task 2 - Headline generation (secondary):
  Given the first paragraph (title_choice_first_paragraph), generate the headline.
  Compare against gold title.
  Metrics: ROUGE-L F1, ChrF++, multilingual SBERT cosine similarity

Models evaluated:
  Baselines (Instruct variants; few-shot prompting):
    - Qwen-0.5B-Instruct
    - Qwen-1.5B-Instruct
    - Qwen-7B-Instruct
  CL fine-tuned (zero-shot prompting):
    - Qwen-0.5B + CL embed-only (paulbontempo/qwen-0.5b-tagalog-cl; trained from base)
    - Qwen-0.5B-Instruct + CL full backprop (paulbontempo/qwen-0.5b-instruct-tagalog-cl-full)
      NOTE: update path after uploading the full-backprop checkpoint to HuggingFace.

Note on dtype: all models loaded in bfloat16. Qwen2.5 is pretrained in bfloat16;
this avoids float16 overflow on larger models and matches training dtype.
"""

import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multilingual SBERT model — loaded once and reused across all model evaluations.
# paraphrase-multilingual-mpnet-base-v2 covers Filipino and is the standard
# multilingual SBERT model for semantic similarity tasks.
print("Loading multilingual SBERT model...")
SBERT_MODEL = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# --- Model configs ---
# Baselines use Instruct variants for better prompt-following.
# CL embed-only was trained from the base model (noted limitation).
# CL full-backprop was trained from Instruct for direct comparability.
# TODO: update qwen_0.5b_cl_full path after uploading checkpoint to HuggingFace.
MODELS = {
    "qwen_0.5b_baseline": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_0.5b_cl_embed": "paulbontempo/qwen-0.5b-tagalog-cl",
    "qwen_0.5b_cl_full": "paulbontempo/qwen-0.5b-instruct-tagalog-cl-full",
    "qwen_1.5b_baseline": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_7b_baseline": "Qwen/Qwen2.5-7B-Instruct",
}

MAX_NEW_TOKENS = 32   # headlines are short
MAX_ARTICLE_TOKENS = 512
MAX_PERPLEXITY_TOKENS = 256
MAX_SAMPLES_PERPLEXITY = 500   # subsample for speed
MAX_SAMPLES_HEADLINE = 200

FEW_SHOT_EXAMPLES = [
    {
        "paragraph": (
            "Inanunsyo ng Pangulo ang bagong programa para sa mga magsasaka sa Mindanao. "
            "Kasama sa programa ang libreng binhi at pataba para sa mga benepisyaryo. "
            "Tinatayang aabot sa sampung libong pamilya ang makikinabang sa inisyatibang ito."
        ),
        "headline": "Naglunsad ang Pangulo ng programa para sa sampung libong magsasaka sa Mindanao",
    },
    {
        "paragraph": (
            "Nagtala ang Bangko Sentral ng Pilipinas ng pagtaas ng inflation rate noong nakaraang buwan. "
            "Ayon sa ulat, umabot ito sa 4.2 porsyento mula sa 3.8 porsyento. "
            "Pangunahing dahilan nito ang pagtaas ng presyo ng pagkain at gasolina."
        ),
        "headline": "Tumaas ang inflation rate sa 4.2% ayon sa Bangko Sentral ng Pilipinas",
    },
]


# ---------------------------------------------------------------------------
# Task 1: Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, texts, max_tokens=MAX_PERPLEXITY_TOKENS, desc=""):
    """Compute mean perplexity over a list of texts using NLL."""
    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc=desc or "Perplexity"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        input_ids = inputs["input_ids"].to(model.device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        # outputs.loss is mean NLL per token
        total_nll += outputs.loss.item() * (seq_len - 1)
        total_tokens += seq_len - 1

    if total_tokens == 0:
        return float("inf")
    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


# ---------------------------------------------------------------------------
# Task 2: Headline generation
# ---------------------------------------------------------------------------

def build_headline_prompt(paragraph, few_shot=True):
    instruction = (
        "Batay sa sumusunod na unang talata ng isang balita, "
        "sumulat ng maikling pamagat (headline) sa Filipino."
    )
    prompt = f"Instruksyon: {instruction}\n\n"

    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            prompt += f"Talata: {ex['paragraph']}\nPamagat: {ex['headline']}\n\n"

    prompt += f"Talata: {paragraph}\nPamagat:"
    return prompt


def generate_headline(model, tokenizer, paragraph, few_shot=True):
    prompt = build_headline_prompt(paragraph, few_shot=few_shot)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ARTICLE_TOKENS,
    )
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
    headline = tokenizer.decode(generated, skip_special_tokens=True).strip()
    headline = headline.split("\n")[0].strip()
    return headline


def compute_generation_metrics(predictions, references):
    """Compute ChrF++, ROUGE-L, and multilingual SBERT cosine similarity."""
    chrf = CHRF(word_order=2)
    chrf_score = chrf.corpus_score(predictions, [references]).score

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    # Multilingual SBERT cosine similarity — semantic similarity in embedding space.
    # Uses the global SBERT_MODEL loaded at startup to avoid reloading per evaluation.
    pred_embs = SBERT_MODEL.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    ref_embs = SBERT_MODEL.encode(references, convert_to_tensor=True, show_progress_bar=False)
    sbert_sims = cos_sim(pred_embs, ref_embs).diagonal()
    sbert_score = float(sbert_sims.mean().item())

    return {"chrf_pp": chrf_score, "rouge_l": rouge_l, "sbert_cos": sbert_score}


# ---------------------------------------------------------------------------
# Model loading + evaluation
# ---------------------------------------------------------------------------

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # bfloat16: matches Qwen2.5 training dtype, avoids float16 overflow on large models,
    # same memory footprint. Safe for all three sizes (0.5B / 1.5B / 7B).
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate_model(model_key, model_path, ppl_texts, headline_paragraphs, headline_titles):
    print("=" * 60)
    print(f"Evaluating: {model_key}")
    print("=" * 60)

    model, tokenizer = load_model(model_path)

    # Task 1: Perplexity
    print("  Task 1: Perplexity...")
    ppl = compute_perplexity(
        model, tokenizer, ppl_texts, desc=f"Perplexity ({model_key})"
    )
    print(f"  Perplexity: {ppl:.2f}")

    # Task 2: Headline generation
    # CL models use zero-shot; baselines use few-shot
    few_shot = "cl" not in model_key
    mode = "few-shot" if few_shot else "zero-shot"
    print(f"  Task 2: Headline generation ({mode})...")
    predictions = []
    for para in tqdm(headline_paragraphs, desc=f"Headlines ({model_key})"):
        pred = generate_headline(model, tokenizer, para, few_shot=few_shot)
        predictions.append(pred)

    gen_metrics = compute_generation_metrics(predictions, list(headline_titles))
    print(f"  ChrF++:       {gen_metrics['chrf_pp']:.2f}")
    print(f"  ROUGE-L:      {gen_metrics['rouge_l']:.4f}")
    print(f"  SBERT cos:    {gen_metrics['sbert_cos']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"perplexity": ppl, **gen_metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading BalitaNLP dataset (LanceBunag/BalitaNLP, no-image config)...")
    ds = load_dataset("LanceBunag/BalitaNLP", name="no-image", split="test")
    print(f"Test split size: {len(ds)}")

    # Task 1: perplexity texts — join body paragraphs into a single string
    ppl_subset = ds.select(range(min(MAX_SAMPLES_PERPLEXITY, len(ds))))
    ppl_texts = [
        " ".join(row["body"]) if isinstance(row["body"], list) else row["body"]
        for row in ppl_subset
    ]

    # Task 2: headline generation — first paragraph -> title
    hl_subset = ds.select(range(min(MAX_SAMPLES_HEADLINE, len(ds))))
    headline_paragraphs = [row["title_choice_first_paragraph"] for row in hl_subset]
    headline_titles = [row["title"] for row in hl_subset]

    results = {}
    for model_key, model_path in MODELS.items():
        try:
            results[model_key] = evaluate_model(
                model_key, model_path,
                ppl_texts, headline_paragraphs, headline_titles,
            )
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
            results[model_key] = None

    print("\n" + "=" * 82)
    print("SUMMARY: BalitaNLP Generation Task Results")
    print("=" * 82)
    print(f"{'Model':<35} {'PPL':>8} {'ChrF++':>8} {'ROUGE-L':>8} {'SBERT cos':>10}")
    print("-" * 82)
    for key, res in results.items():
        if res:
            print(
                f"  {key:<33} "
                f"{res['perplexity']:>8.2f} "
                f"{res['chrf_pp']:>8.2f} "
                f"{res['rouge_l']:>8.4f} "
                f"{res['sbert_cos']:>10.4f}"
            )
        else:
            print(f"  {key:<33} {'FAILED':>8}")
