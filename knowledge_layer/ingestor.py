"""
ingestor.py — Document ingestion + chunking pipeline

Supports: PDF, plain text (.txt), and Markdown (.md) files.
Splits documents into overlapping chunks ready for embedding.

Usage:
    from knowledge_layer.ingestor import ingest_documents
    chunks = ingest_documents("data/raw_docs")
"""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import RAGConfig


# ── Supported file extensions ──────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _load_file(file_path: Path) -> List[Document]:
    """Load a single file and return a list of LangChain Documents."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        logger.warning(f"Skipping unsupported file type: {file_path}")
        return []

    docs = loader.load()
    # Tag each doc with source filename for traceability
    for doc in docs:
        doc.metadata["source"] = file_path.name
    logger.info(f"Loaded {len(docs)} page(s) from {file_path.name}")
    return docs


def load_all_documents(docs_dir: str) -> List[Document]:
    """
    Walk docs_dir and load all supported files.

    Args:
        docs_dir: Path to folder containing raw documents.

    Returns:
        Flat list of LangChain Document objects (one per page/section).
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    all_docs: List[Document] = []
    files = [f for f in docs_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning(f"No supported documents found in {docs_dir}")
        return []

    for file_path in sorted(files):
        all_docs.extend(_load_file(file_path))

    logger.info(f"Total pages/sections loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Chunk size and overlap come from RAGConfig (set in .env):
        CHUNK_SIZE=350, CHUNK_OVERLAP=50

    Args:
        documents: Raw LangChain Document objects.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.chunk_size,
        chunk_overlap=RAGConfig.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # natural break points first
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata for debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info(
        f"Created {len(chunks)} chunks "
        f"(size={RAGConfig.chunk_size}, overlap={RAGConfig.chunk_overlap})"
    )
    return chunks


def ingest_documents(docs_dir: str = "data/raw_docs") -> List[Document]:
    """
    Full ingestion pipeline: load files → chunk.

    Args:
        docs_dir: Folder with raw documents (PDFs, txt, md).

    Returns:
        Ready-to-embed list of Document chunks.
    """
    logger.info(f"Starting ingestion from: {docs_dir}")
    raw_docs = load_all_documents(docs_dir)

    if not raw_docs:
        logger.warning("No documents loaded — nothing to chunk.")
        return []

    chunks = chunk_documents(raw_docs)
    logger.info(f"Ingestion complete: {len(chunks)} chunks ready for embedding.")
    return chunks


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_docs"
    chunks = ingest_documents(docs_dir)
    print(f"\n✓ {len(chunks)} chunks created.")
    if chunks:
        print(f"\nSample chunk [0]:\n{chunks[0].page_content[:300]}")
        print(f"Metadata: {chunks[0].metadata}")
