# ============================================================
# CareSync AI — Q&A Generation on Kaggle GPU
# Run this as a Kaggle notebook to generate qa_pairs.jsonl fast
#
# Steps:
# 1. Upload your raw_docs PDFs as a Kaggle dataset
# 2. Create a new Kaggle notebook, enable GPU T4 + Internet ON
# 3. Paste this entire script and run all cells
# ============================================================

# ── CELL 1: Install ───────────────────────────────────────────
# !pip install -q transformers pypdf tqdm accelerate

# ── CELL 2: Imports ───────────────────────────────────────────
import json, re, os, torch
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── CELL 3: Config ────────────────────────────────────────────
DOCS_DIR    = "/kaggle/input/caresync-docs/"   # update to your dataset path
OUTPUT_FILE = "/kaggle/working/qa_pairs.jsonl"
TARGET      = 300
CHUNK_SIZE  = 350
CHUNK_OVERLAP = 50
MODEL_ID    = "microsoft/Phi-3-mini-4k-instruct"

# ── CELL 4: Load + chunk PDFs ─────────────────────────────────
def load_pdfs(docs_dir):
    docs = []
    for f in sorted(Path(docs_dir).glob("*.pdf")):
        reader = PdfReader(str(f))
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        docs.append((f.name, text))
        print(f"Loaded: {f.name} ({len(reader.pages)} pages)")
    return docs

def split_text(text, chunk_size=350, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

docs = load_pdfs(DOCS_DIR)
all_chunks = []
for source, text in docs:
    for i, chunk in enumerate(split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
        all_chunks.append({"text": chunk, "source": source, "chunk_id": i})

print(f"\nTotal chunks: {len(all_chunks)}")

# Only use enough chunks to hit target (target/2 * 1.3 buffer)
max_chunks = int((TARGET / 2) * 1.3)
chunks_to_use = all_chunks[:max_chunks]
print(f"Using first {len(chunks_to_use)} chunks to generate {TARGET} pairs")

# ── CELL 5: Load Phi-3 mini (fp16, no quantization needed on T4) ──
print("\nLoading Phi-3 mini (fp16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model ready.")

# ── CELL 6: Q&A generation function ──────────────────────────
SYSTEM = 'Output ONLY valid JSON, no explanation. Format: [{"question":"...","answer":"..."},{"question":"...","answer":"..."}]'

def generate_qa(chunk_text):
    prompt = (
        f"<|system|>\n{SYSTEM}<|end|>\n"
        f"<|user|>\nPassage: {chunk_text[:600]}\n\nGive 2 medical Q&A pairs as JSON:<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Extract JSON array robustly — handles extra text around the array
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        pairs = json.loads(match.group())
        return [
            {"question": p["question"].strip(), "answer": p["answer"].strip()}
            for p in pairs
            if isinstance(p, dict)
            and len(p.get("question", "").strip()) > 10
            and len(p.get("answer", "").strip()) > 10
        ]
    except Exception:
        return []

# ── CELL 7: Run generation ────────────────────────────────────
all_pairs = []
failed = 0

with open(OUTPUT_FILE, "w") as f:
    for chunk in tqdm(chunks_to_use, desc="Generating Q&A"):
        if len(all_pairs) >= TARGET:
            break
        pairs = generate_qa(chunk["text"])
        if not pairs:
            failed += 1
        for pair in pairs:
            record = {
                "question": pair["question"],
                "answer":   pair["answer"],
                "source":   chunk["source"],
                "chunk_id": chunk["chunk_id"],
            }
            f.write(json.dumps(record) + "\n")
            all_pairs.append(record)
            if len(all_pairs) >= TARGET:
                break

print(f"\n✓ Generated {len(all_pairs)} Q&A pairs")
print(f"  Failed chunks: {failed}")
print(f"  Saved to: {OUTPUT_FILE}")

# ── CELL 8: Preview first 5 pairs ────────────────────────────
print("\nFirst 5 pairs:\n")
for i, p in enumerate(all_pairs[:5], 1):
    print(f"[{i}] Q: {p['question']}")
    print(f"    A: {p['answer']}")
    print(f"    Src: {p['source']}\n")

print("Download qa_pairs.jsonl from Kaggle output tab → save to data/qa_pairs/")
