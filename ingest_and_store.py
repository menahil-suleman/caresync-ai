"""
ingest_and_store.py — One-shot Day 1 runner

Run this script to:
  1. Ingest all documents from data/raw_docs
  2. Embed them and store in pgvector
  3. (Optional) Generate Q&A pairs for Day 2 fine-tuning

Usage:
    # Ingest + store only:
    python ingest_and_store.py

    # Ingest + store + generate Q&A:
    python ingest_and_store.py --generate-qa

    # Custom docs directory:
    python ingest_and_store.py --docs-dir path/to/docs --generate-qa
"""

import argparse
import sys

from loguru import logger

from knowledge_layer.embedder import store_chunks
from knowledge_layer.ingestor import ingest_documents
from knowledge_layer.qa_generator import generate_qa_dataset


def main():
    parser = argparse.ArgumentParser(description="CareSync AI — Ingest & Store Pipeline")
    parser.add_argument(
        "--docs-dir",
        default="data/raw_docs",
        help="Directory containing raw documents (default: data/raw_docs)",
    )
    parser.add_argument(
        "--generate-qa",
        action="store_true",
        help="Also generate Q&A pairs for fine-tuning after ingestion",
    )
    args = parser.parse_args()

    # ── Step 1: Ingest & chunk ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Document Ingestion + Chunking")
    logger.info("=" * 60)
    chunks = ingest_documents(args.docs_dir)

    if not chunks:
        logger.error("No chunks produced. Add documents to data/raw_docs and retry.")
        sys.exit(1)

    # ── Step 2: Embed & store in pgvector ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Embedding + pgvector Storage")
    logger.info("=" * 60)
    store_chunks(chunks)

    # ── Step 3 (optional): Q&A generation ─────────────────────────────────────
    if args.generate_qa:
        logger.info("=" * 60)
        logger.info("STEP 3: Q&A Pair Generation for Fine-Tuning")
        logger.info("=" * 60)
        output_path = generate_qa_dataset(args.docs_dir)
        logger.info(f"Q&A pairs saved to: {output_path}")

    logger.info("=" * 60)
    logger.info("All done! RAG pipeline is ready.")
    logger.info("Test it: python -m knowledge_layer.rag_chain")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
