"""
qa_generator.py — Generate 200-500 Q&A pairs from chunks (zero LangChain)

Output: JSONL file at data/qa_pairs/qa_pairs.jsonl
Each line: {"question": "...", "answer": "...", "source": "...", "chunk_id": 0}

Used on Day 2 for LoRA fine-tuning of Phi-3 mini.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List

import ollama as ollama_client
from loguru import logger
from tqdm import tqdm

from config import OllamaConfig, QAConfig
from knowledge_layer.ingestor import Chunk, ingest_documents


# ── Prompt — short and strict, forces clean JSON ───────────────────────────────
_SYSTEM = 'Output ONLY valid JSON, no explanation, no markdown. Format: [{"question":"...","answer":"..."},{"question":"...","answer":"..."}]'

_USER = "Passage: {text}\n\nGive 2 medical Q&A pairs as JSON array:"


def _extract_json(raw: str) -> list:
    """
    Robustly extract a JSON array from model output.
    Handles markdown fences, extra text before/after the array.
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find first [ ... ] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No JSON array found", raw, 0)


def _generate_pairs(chunk_text: str, retries: int = 2) -> List[dict]:
    """Generate Q&A pairs for one chunk. Returns [] on failure."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER.format(text=chunk_text[:800])},
    ]

    for attempt in range(retries + 1):
        try:
            resp = ollama_client.chat(
                model=OllamaConfig.model,
                messages=messages,
                options={
                    "temperature": 0.3,
                    "num_predict": 300,
                    "stop": ["\n\n", "Passage:"],  # stop early if model rambles
                },
            )
            raw = resp["message"]["content"].strip()
            pairs = _extract_json(raw)

            validated = [
                {"question": p["question"].strip(), "answer": p["answer"].strip()}
                for p in pairs
                if isinstance(p, dict)
                and "question" in p and "answer" in p
                and len(p["question"].strip()) > 10
                and len(p["answer"].strip()) > 10
            ]
            return validated

        except (json.JSONDecodeError, KeyError, TypeError):
            if attempt < retries:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}/{retries + 1}), retrying...")
                time.sleep(0.5)
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
    Generate Q&A pairs from documents.

    Stops as soon as target_count pairs are collected — no need to process
    all 3413 chunks, we just need 300 good pairs for fine-tuning.
    """
    out_path = Path(output_file or QAConfig.output_file)
    target   = target_count or QAConfig.target_count
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = ingest_documents(docs_dir)
    if not chunks:
        logger.error("No chunks to generate Q&A from.")
        return str(out_path)

    # Only process as many chunks as needed — target/2 chunks = target pairs
    # Add 20% buffer for failed generations
    max_chunks = int((target / 2) * 1.2)
    chunks_to_process = chunks[:max_chunks]

    logger.info(f"Generating up to {target} Q&A pairs from "
                f"{len(chunks_to_process)}/{len(chunks)} chunks...")

    all_pairs: List[dict] = []

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks_to_process, desc="Q&A generation"):
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
