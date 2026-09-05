"""
qa_generator.py — Generate 200–500 Q&A pairs from document chunks

Strategy:
  For each chunk, prompt Ollama to generate 1–2 question/answer pairs.
  Output is saved as JSONL at data/qa_pairs/qa_pairs.jsonl

Each line in the JSONL file:
    {"question": "...", "answer": "...", "source": "...", "chunk_id": 0}

This file is used on Day 2 for LoRA fine-tuning of Phi-3 mini.

Usage:
    python -m knowledge_layer.qa_generator
    # or with custom doc dir:
    python -m knowledge_layer.qa_generator data/raw_docs
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List

import ollama as ollama_client
from loguru import logger
from tqdm import tqdm

from config import OllamaConfig, QAConfig
from knowledge_layer.ingestor import ingest_documents

# ── Prompt for Q&A generation ─────────────────────────────────────────────────
QA_GEN_SYSTEM = """You are a medical education expert creating training data for a health AI assistant.
Given a passage of health-related text, generate exactly 2 question-answer pairs.

STRICT FORMAT — respond with valid JSON only, no extra text:
[
  {"question": "<specific question answerable from the passage>", "answer": "<concise accurate answer based only on the passage>"},
  {"question": "<another specific question>", "answer": "<another concise answer>"}
]

Rules:
- Questions must be answerable from the passage alone
- Answers must be factual and grounded in the passage
- Do not generate generic or vague questions
- Focus on clinically useful information (symptoms, values, conditions, advice)
"""

QA_GEN_USER = """Passage:
{chunk_text}

Generate 2 question-answer pairs as JSON:"""


def generate_qa_from_chunk(chunk_text: str, retries: int = 2) -> List[dict]:
    """
    Ask Ollama to generate Q&A pairs from a single chunk.

    Args:
        chunk_text: The document chunk text.
        retries:    Number of retry attempts on parse failure.

    Returns:
        List of dicts with keys: question, answer.
        Returns empty list if generation fails.
    """
    messages = [
        {"role": "system", "content": QA_GEN_SYSTEM},
        {"role": "user", "content": QA_GEN_USER.format(chunk_text=chunk_text[:1200])},
    ]

    for attempt in range(retries + 1):
        try:
            response = ollama_client.chat(
                model=OllamaConfig.model,
                messages=messages,
                options={"temperature": 0.4, "num_predict": 400},
            )
            raw = response["message"]["content"].strip()

            # Extract JSON — handle cases where model wraps in markdown
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            pairs = json.loads(raw)

            # Validate structure
            validated = []
            for pair in pairs:
                if "question" in pair and "answer" in pair:
                    q = str(pair["question"]).strip()
                    a = str(pair["answer"]).strip()
                    if len(q) > 10 and len(a) > 10:
                        validated.append({"question": q, "answer": a})

            return validated

        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Ollama error during Q&A gen: {e}")
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
        output_file:  Path to save .jsonl output.
        target_count: Stop after generating this many pairs.

    Returns:
        Path to the saved JSONL file.
    """
    output_path = Path(output_file or QAConfig.output_file)
    target = target_count or QAConfig.target_count
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Q&A generation — target: {target} pairs")
    chunks = ingest_documents(docs_dir)

    if not chunks:
        logger.error("No chunks to generate Q&A from.")
        return str(output_path)

    all_pairs: List[dict] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Generating Q&A pairs"):
            if len(all_pairs) >= target:
                break

            pairs = generate_qa_from_chunk(chunk.page_content)

            for pair in pairs:
                record = {
                    "question":  pair["question"],
                    "answer":    pair["answer"],
                    "source":    chunk.metadata.get("source", "unknown"),
                    "chunk_id":  chunk.metadata.get("chunk_id", -1),
                }
                f.write(json.dumps(record) + "\n")
                all_pairs.append(record)

                if len(all_pairs) >= target:
                    break

    logger.info(
        f"Q&A generation complete — {len(all_pairs)} pairs saved to {output_path}"
    )
    return str(output_path)


def load_qa_pairs(jsonl_path: str) -> List[dict]:
    """Load previously generated Q&A pairs from JSONL."""
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
    output = generate_qa_dataset(docs_dir)
    print(f"\n✓ Dataset saved to: {output}")

    # Preview first 3 pairs
    pairs = load_qa_pairs(output)
    print(f"\nPreview (first 3 of {len(pairs)}):")
    for i, p in enumerate(pairs[:3], 1):
        print(f"\n[{i}] Q: {p['question']}")
        print(f"    A: {p['answer']}")
        print(f"    Source: {p['source']}")
