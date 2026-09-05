"""
finetune_phi3.py — LoRA fine-tuning of Phi-3 mini on CareSync Q&A pairs

Run this on Kaggle (free T4 GPU) or Google Colab.

Pipeline:
  1. Load Q&A pairs from JSONL
  2. Format into Phi-3 chat template
  3. Load Phi-3 mini in 4-bit (QLoRA) via bitsandbytes
  4. Attach LoRA adapters via PEFT
  5. Optuna sweep over lr, lora_rank, epochs
  6. Train best config, evaluate vs baseline
  7. Save fine-tuned adapter

Install on Kaggle/Colab (run in a cell first):
    !pip install -q transformers peft bitsandbytes optuna datasets tqdm loguru
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict

# ── Imports (all available on Kaggle/Colab) ────────────────────────────────────
import optuna
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL      = "microsoft/Phi-3-mini-4k-instruct"
QA_FILE         = "qa_pairs.jsonl"          # upload this to Kaggle
OUTPUT_DIR      = "./phi3_caresync_lora"
BASELINE_RESULTS_FILE = "baseline_results.json"
FINETUNED_RESULTS_FILE = "finetuned_results.json"

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Data loading + formatting ───────────────────────────────────────────────

def load_qa_pairs(jsonl_path: str) -> List[Dict]:
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs)} Q&A pairs from {jsonl_path}")
    return pairs


def format_phi3_prompt(question: str, answer: str) -> str:
    """
    Phi-3 instruct chat template format.
    <|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>
    """
    return (
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n{answer}<|end|>"
    )


def prepare_dataset(pairs: List[Dict], tokenizer, max_length: int = 512):
    """
    Tokenise Q&A pairs into model inputs.
    Labels are set to -100 for the prompt (user) portion — we only
    train the model to predict the assistant answer.
    """
    formatted = [
        format_phi3_prompt(p["question"], p["answer"])
        for p in pairs
    ]

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()

        # Mask prompt tokens so loss is only on the answer
        for i, text in enumerate(batch["text"]):
            prompt_end = text.find("<|assistant|>") + len("<|assistant|>\n")
            prompt_tokens = tokenizer(
                text[:prompt_end], return_tensors="pt"
            )["input_ids"].shape[1]
            tokens["labels"][i, :prompt_tokens] = -100

        return tokens

    dataset = Dataset.from_dict({"text": formatted})
    dataset = dataset.map(tokenize, batched=True, batch_size=8,
                          remove_columns=["text"])
    return dataset


# ── 2. Model loading (4-bit QLoRA) ────────────────────────────────────────────

def load_base_model(quantize: bool = True):
    """
    Load Phi-3 mini with optional 4-bit quantization.
    4-bit lets it fit on a free T4 GPU (15GB VRAM).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,       # extra memory saving
    ) if quantize else None

    logger.info(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Base model loaded.")
    return model, tokenizer


def attach_lora(model, lora_rank: int = 16, lora_alpha: int = 32,
                lora_dropout: float = 0.05):
    """
    Attach LoRA adapters to the attention layers.

    LoRA only trains low-rank matrices (rank r) injected into Q/K/V/O
    projections — updates ~0.1% of parameters instead of 100%.

    Args:
        lora_rank:    Rank of the low-rank matrices (higher = more capacity)
        lora_alpha:   Scaling factor (usually 2x rank)
        lora_dropout: Dropout on LoRA layers
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── 3. Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, test_pairs: List[Dict],
                   label: str = "model") -> Dict:
    """
    Run inference on held-out test questions and compute:
    - Exact match rate (answer appears in generated text)
    - Average generation length

    Returns dict of metrics for comparison.
    """
    model.eval()
    exact_matches = 0
    results = []

    with torch.no_grad():
        for pair in tqdm(test_pairs[:50], desc=f"Evaluating {label}"):
            prompt = f"<|user|>\n{pair['question']}<|end|>\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Simple exact-match: check if key answer words appear
            answer_words = set(pair["answer"].lower().split())
            gen_words = set(generated.lower().split())
            overlap = len(answer_words & gen_words) / max(len(answer_words), 1)

            if overlap > 0.5:
                exact_matches += 1

            results.append({
                "question":  pair["question"],
                "expected":  pair["answer"],
                "generated": generated,
                "overlap":   round(overlap, 3),
            })

    metrics = {
        "label":        label,
        "tested":       len(results),
        "exact_match":  round(exact_matches / len(results), 3),
        "avg_overlap":  round(sum(r["overlap"] for r in results) / len(results), 3),
        "samples":      results[:5],   # save first 5 for inspection
    }
    logger.info(f"{label} — exact_match: {metrics['exact_match']}, "
                f"avg_overlap: {metrics['avg_overlap']}")
    return metrics


# ── 4. Training ────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_dataset, output_dir: str,
          learning_rate: float, num_epochs: int):
    """Fine-tune with HuggingFace Trainer."""
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch = 16
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",        # memory-efficient optimizer
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",                # disable wandb
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        ),
    )
    trainer.train()
    return trainer


# ── 5. Optuna hyperparameter sweep ────────────────────────────────────────────

def optuna_sweep(train_pairs: List[Dict], val_pairs: List[Dict],
                 n_trials: int = 6) -> Dict:
    """
    Sweep over:
      - learning_rate: [1e-4, 3e-4, 1e-3]
      - lora_rank:     [8, 16, 32]
      - num_epochs:    [1, 2, 3]

    Each trial trains a fresh LoRA adapter and evaluates on val_pairs.
    Returns the best hyperparameters found.

    n_trials=6 is enough for a 4-day build — covers the main lr/rank space.
    """
    def objective(trial):
        lr         = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3])
        lora_rank  = trial.suggest_categorical("lora_rank", [8, 16, 32])
        num_epochs = trial.suggest_int("num_epochs", 1, 3)

        logger.info(f"Trial {trial.number}: lr={lr}, rank={lora_rank}, "
                    f"epochs={num_epochs}")

        model, tokenizer = load_base_model()
        model = attach_lora(model, lora_rank=lora_rank,
                            lora_alpha=lora_rank * 2)

        train_ds = prepare_dataset(train_pairs, tokenizer)
        trial_dir = f"./trial_{trial.number}"

        train(model, tokenizer, train_ds, trial_dir, lr, num_epochs)

        metrics = evaluate_model(model, tokenizer, val_pairs,
                                 label=f"trial_{trial.number}")
        # Optuna maximises this score
        return metrics["avg_overlap"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best score:  {study.best_value:.4f}")
    return study.best_params


# ── 6. Main pipeline ───────────────────────────────────────────────────────────

def main():
    # Load data
    pairs = load_qa_pairs(QA_FILE)
    random.shuffle(pairs)

    # Split: 80% train, 10% val, 10% test
    n = len(pairs)
    train_pairs = pairs[:int(n * 0.8)]
    val_pairs   = pairs[int(n * 0.8):int(n * 0.9)]
    test_pairs  = pairs[int(n * 0.9):]
    logger.info(f"Split — train: {len(train_pairs)}, val: {len(val_pairs)}, "
                f"test: {len(test_pairs)}")

    # ── Baseline evaluation (no fine-tuning) ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION (pretrained Phi-3, no fine-tuning)")
    logger.info("=" * 60)
    base_model, tokenizer = load_base_model()
    baseline_metrics = evaluate_model(
        base_model, tokenizer, test_pairs, label="baseline"
    )
    with open(BASELINE_RESULTS_FILE, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    logger.info(f"Baseline saved to {BASELINE_RESULTS_FILE}")

    # Free base model memory before sweep
    del base_model
    torch.cuda.empty_cache()

    # ── Optuna sweep ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("OPTUNA HYPERPARAMETER SWEEP")
    logger.info("=" * 60)
    best_params = optuna_sweep(train_pairs, val_pairs, n_trials=6)

    # ── Final training with best params ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"FINAL TRAINING — best params: {best_params}")
    logger.info("=" * 60)
    model, tokenizer = load_base_model()
    model = attach_lora(
        model,
        lora_rank=best_params["lora_rank"],
        lora_alpha=best_params["lora_rank"] * 2,
    )
    train_ds = prepare_dataset(train_pairs, tokenizer)
    train(model, tokenizer, train_ds, OUTPUT_DIR,
          best_params["lr"], best_params["num_epochs"])

    # Save final adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Fine-tuned adapter saved to {OUTPUT_DIR}")

    # ── Fine-tuned evaluation ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINE-TUNED MODEL EVALUATION")
    logger.info("=" * 60)
    finetuned_metrics = evaluate_model(
        model, tokenizer, test_pairs, label="finetuned"
    )
    with open(FINETUNED_RESULTS_FILE, "w") as f:
        json.dump(finetuned_metrics, f, indent=2)

    # ── Final comparison ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info(f"Baseline  — exact_match: {baseline_metrics['exact_match']}, "
                f"avg_overlap: {baseline_metrics['avg_overlap']}")
    logger.info(f"Fine-tuned — exact_match: {finetuned_metrics['exact_match']}, "
                f"avg_overlap: {finetuned_metrics['avg_overlap']}")
    improvement = finetuned_metrics['avg_overlap'] - baseline_metrics['avg_overlap']
    logger.info(f"Improvement: +{improvement:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
