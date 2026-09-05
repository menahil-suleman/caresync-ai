# ============================================================
# CareSync AI — Day 2: LoRA Fine-tuning on Kaggle
# Run each cell in order on a Kaggle T4 GPU notebook
# ============================================================

# ── CELL 1: Install dependencies ──────────────────────────────
# !pip install -q transformers peft bitsandbytes optuna datasets tqdm loguru accelerate

# ── CELL 2: Upload your qa_pairs.jsonl ────────────────────────
# In Kaggle: Add Data → Upload → select qa_pairs.jsonl
# It will be at: /kaggle/input/your-dataset/qa_pairs.jsonl
# Update QA_FILE path below accordingly

# ── CELL 3: Imports ───────────────────────────────────────────
import json, os, random, torch
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
import optuna
from tqdm import tqdm

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── CELL 4: Config ────────────────────────────────────────────
BASE_MODEL  = "microsoft/Phi-3-mini-4k-instruct"
QA_FILE     = "/kaggle/input/caresync-qa/qa_pairs.jsonl"  # update path
OUTPUT_DIR  = "/kaggle/working/phi3_caresync_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CELL 5: Load Q&A pairs ────────────────────────────────────
def load_qa_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} Q&A pairs")
    return pairs

pairs = load_qa_pairs(QA_FILE)
random.shuffle(pairs)

n = len(pairs)
train_pairs = pairs[:int(n * 0.8)]
val_pairs   = pairs[int(n * 0.8):int(n * 0.9)]
test_pairs  = pairs[int(n * 0.9):]
print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

# ── CELL 6: Format + tokenise ─────────────────────────────────
def format_prompt(q, a):
    return f"<|user|>\n{q}<|end|>\n<|assistant|>\n{a}<|end|>"

def prepare_dataset(pairs, tokenizer, max_length=512):
    texts = [format_prompt(p["question"], p["answer"]) for p in pairs]

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"], truncation=True,
            max_length=max_length, padding="max_length",
        )
        labels = list(tokens["input_ids"])
        # Mask prompt — only train on answer tokens
        for i, text in enumerate(batch["text"]):
            prompt_end = text.find("<|assistant|>") + len("<|assistant|>\n")
            prompt_len = len(tokenizer(text[:prompt_end])["input_ids"])
            labels[i] = [-100] * prompt_len + labels[i][prompt_len:]
        tokens["labels"] = labels
        return tokens

    ds = Dataset.from_dict({"text": texts})
    return ds.map(tokenize, batched=True, batch_size=8, remove_columns=["text"])

# ── CELL 7: Load base model (4-bit QLoRA) ─────────────────────
def load_model_and_tokenizer():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ── CELL 8: Baseline evaluation (no fine-tuning) ──────────────
def evaluate(model, tokenizer, pairs, label="model", n=50):
    model.eval()
    hits, results = 0, []
    with torch.no_grad():
        for p in tqdm(pairs[:n], desc=f"Eval {label}"):
            prompt = f"<|user|>\n{p['question']}<|end|>\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=200,
                temperature=0.1, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            overlap = len(
                set(p["answer"].lower().split()) & set(gen.lower().split())
            ) / max(len(p["answer"].lower().split()), 1)

            if overlap > 0.5: hits += 1
            results.append({"q": p["question"], "expected": p["answer"],
                             "generated": gen, "overlap": overlap})

    metrics = {
        "label": label,
        "exact_match": round(hits / len(results), 3),
        "avg_overlap": round(sum(r["overlap"] for r in results) / len(results), 3),
    }
    print(f"\n{label} — exact_match: {metrics['exact_match']}, "
          f"avg_overlap: {metrics['avg_overlap']}")
    return metrics, results

print("Loading base model for baseline evaluation...")
base_model, tokenizer = load_model_and_tokenizer()
baseline_metrics, baseline_results = evaluate(base_model, tokenizer, test_pairs, "baseline")
print("Baseline metrics:", baseline_metrics)

# ── CELL 9: Optuna sweep ──────────────────────────────────────
# Free baseline model memory first
del base_model
torch.cuda.empty_cache()

def objective(trial):
    lr        = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3])
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32])
    epochs    = trial.suggest_int("epochs", 1, 3)

    print(f"\nTrial {trial.number}: lr={lr}, rank={lora_rank}, epochs={epochs}")

    model, tok = load_model_and_tokenizer()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_rank,
        lora_alpha=lora_rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = prepare_dataset(train_pairs, tok)
    args = TrainingArguments(
        output_dir=f"./trial_{trial.number}", num_train_epochs=epochs,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=lr, fp16=True, logging_steps=10,
        save_strategy="no", optim="paged_adamw_8bit",
        lr_scheduler_type="cosine", warmup_ratio=0.05,
        report_to="none", seed=SEED,
    )
    Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt"),
    ).train()

    m, _ = evaluate(model, tok, val_pairs, f"trial_{trial.number}", n=30)

    del model
    torch.cuda.empty_cache()
    return m["avg_overlap"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=6)
print(f"\nBest params: {study.best_params}")
print(f"Best score:  {study.best_value:.4f}")

# ── CELL 10: Final training with best params ──────────────────
best = study.best_params
print(f"\nFinal training — {best}")

final_model, tokenizer = load_model_and_tokenizer()
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=best["lora_rank"],
    lora_alpha=best["lora_rank"] * 2, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
)
final_model = get_peft_model(final_model, lora_cfg)

full_train = pairs[:int(n * 0.9)]   # train + val for final run
ds_final = prepare_dataset(full_train, tokenizer)

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=best["epochs"],
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=best["lr"], fp16=True, logging_steps=10,
    save_strategy="epoch", optim="paged_adamw_8bit",
    lr_scheduler_type="cosine", warmup_ratio=0.05,
    report_to="none", seed=SEED,
)
Trainer(
    model=final_model, args=args, train_dataset=ds_final,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
).train()

final_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# ── CELL 11: Final comparison ─────────────────────────────────
finetuned_metrics, finetuned_results = evaluate(
    final_model, tokenizer, test_pairs, "finetuned"
)

print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)
print(f"Baseline   — exact_match: {baseline_metrics['exact_match']:.3f} | "
      f"avg_overlap: {baseline_metrics['avg_overlap']:.3f}")
print(f"Fine-tuned — exact_match: {finetuned_metrics['exact_match']:.3f} | "
      f"avg_overlap: {finetuned_metrics['avg_overlap']:.3f}")
print(f"Improvement: +{finetuned_metrics['avg_overlap'] - baseline_metrics['avg_overlap']:.4f}")
print("="*50)

# Save results
with open(f"{OUTPUT_DIR}/baseline_results.json", "w") as f:
    json.dump({"metrics": baseline_metrics, "samples": baseline_results[:10]}, f, indent=2)
with open(f"{OUTPUT_DIR}/finetuned_results.json", "w") as f:
    json.dump({"metrics": finetuned_metrics, "samples": finetuned_results[:10]}, f, indent=2)

print("\nDone! Download phi3_caresync_lora/ folder from Kaggle output.")
print("Copy it to language_layer/phi3_caresync_lora/ in your local repo.")
