"""
qa_generator.py — Generate 200–500 Q&A pairs from chunks (zero LangChain)

Output: JSONL file at data/qa_pairs/qa_pairs.jsonl
Each line: {"question": "...", "answer": "...", "source": "...", "chunk_id": 0}

Used on Day 2 for LoRA fine-tuning of Phi-3 mini.

Usage:
    python -m knowledge_layer.qa_generator
    python -m knowledge_layer.qa_generator data/raw_docs
"""

import json
import sys
import time
from pathlib import Path
from typing import List

import ollama as ollama_client
from loguru import logger
from tqdm import tqdm

from config import OllamaConfig, QAConfig
from knowledge_layer.ingestor import Chunk, ingest_documents


# ── Prompt ─────────────────────────────────────────────────────────────────────
_SYSTEM = """You are a medical education expert creating training data for a health AI.
Given a health-related passage, generate exactly 2 question-answer pairs.

Respond with valid JSON only — no extra text, no markdown:
[
  {"question": "<specific question answerable from the passage>", "answer": "<concise factual answer>"},
  {"question": "<another specific question>", "answer": "<another concise answer>"}
]

Rules:
- Questions must be answerable from the passage alone
- Answers must be grounded in the passage — no outside knowledge
- Focus on clinically useful facts (normal ranges, symptoms, conditions, advice)
- No vague or generic questions"""

_USER = "Passage:\n{text}\n\nGenerate 2 Q&A pairs as JSON:"


def _generate_pairs(chunk_text: str, retries: int = 2) -> List[dict]:
    """Ask Ollama to produce Q&A pairs for one chunk. Returns [] on failure."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER.format(text=chunk_text[:1200])},
    ]

    for attempt in range(retries + 1):
        try:
            resp = ollama_client.chat(
                model=OllamaConfig.model,
                messages=messages,
                options={"temperature": 0.4, "num_predict": 400},
            )
            raw = resp["message"]["content"].strip()

            # Strip markdown code fences if model adds them
            if "```" in raw:
                raw = raw.split("```")[1]
                raw = raw.lstrip("json").strip()

            pairs = json.loads(raw)
            validated = [
                {"question": p["question"].strip(), "answer": p["answer"].strip()}
                for p in pairs
                if "question" in p and "answer" in p
                and len(p["question"].strip()) > 10
                and len(p["answer"].strip()) > 10
            ]
            return validated

        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed (attempt {attempt + 1}/{retries + 1})")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return []

    return []


def generate_qa_dataset(
    docs_dir: str = "data/raw_docs",
    output_file: str = None,
    target_count: int = None,
) -> str:
    """
    Full Q&A generation pipeline.

    Args:
        docs_dir:     Folder with raw documents.
        output_file:  Where to save the JSONL file.
        target_count: Stop after this many pairs.

    Returns:
        Path to the saved JSONL file.
    """
    out_path = Path(output_file or QAConfig.output_file)
    target   = target_count or QAConfig.target_count
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = ingest_documents(docs_dir)
    if not chunks:
        logger.error("No chunks to generate Q&A from.")
        return str(out_path)

    logger.info(f"Generating up to {target} Q&A pairs from {len(chunks)} chunks...")
    all_pairs: List[dict] = []

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Q&A generation"):
            if len(all_pairs) >= target:
                break

            for pair in _generate_pairs(chunk.text):
                record = {
                    "question": pair["question"],
                    "answer":   pair["answer"],
                    "source":   chunk.source,
                    "chunk_id": chunk.chunk_id,
                }
                f.write(json.dumps(record) + "\n")
                all_pairs.append(record)
                if len(all_pairs) >= target:
                    break

    logger.info(f"Done — {len(all_pairs)} pairs saved to {out_path}")
    return str(out_path)


def load_qa_pairs(jsonl_path: str) -> List[dict]:
    """Load a previously generated Q&A JSONL file."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs)} Q&A pairs from {jsonl_path}")
    return pairs


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_docs"
    out = generate_qa_dataset(docs_dir)
    print(f"\n✓ Saved to: {out}")
    pairs = load_qa_pairs(out)
    print(f"\nPreview (first 3 of {len(pairs)}):")
    for i, p in enumerate(pairs[:3], 1):
        print(f"\n[{i}] Q: {p['question']}\n    A: {p['answer']}\n    Src: {p['source']}")
