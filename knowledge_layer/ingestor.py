"""
ingestor.py — Document ingestion + chunking (zero LangChain)

Supports: PDF, plain text (.txt), Markdown (.md)
Splits text into overlapping chunks using a pure-Python recursive splitter.

Public API:
    ingest_documents(docs_dir) -> List[Chunk]

Each Chunk is a dataclass:
    Chunk(text, source, chunk_id, chunk_size)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader
from loguru import logger

from config import RAGConfig


# ── Data model (replaces LangChain Document) ──────────────────────────────────
@dataclass
class Chunk:
    text:       str
    source:     str          # filename
    chunk_id:   int          # index within the source doc
    chunk_size: int = field(init=False)

    def __post_init__(self):
        self.chunk_size = len(self.text)


# ── File loaders ──────────────────────────────────────────────────────────────
def _load_pdf(path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    logger.info(f"PDF loaded: {path.name} ({len(reader.pages)} pages)")
    return text


def _load_text(path: Path) -> str:
    """Load plain text or markdown file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info(f"Text loaded: {path.name} ({len(text)} chars)")
    return text


LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md":  _load_text,
}


def load_all_documents(docs_dir: str) -> List[tuple[str, str]]:
    """
    Walk docs_dir, load every supported file.

    Returns:
        List of (filename, full_text) tuples.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    results = []
    files = [f for f in sorted(docs_path.iterdir()) if f.suffix.lower() in LOADERS]

    if not files:
        logger.warning(f"No supported files found in {docs_dir}")
        return []

    for f in files:
        try:
            text = LOADERS[f.suffix.lower()](f)
            if text.strip():
                results.append((f.name, text))
        except Exception as e:
            logger.error(f"Failed to load {f.name}: {e}")

    logger.info(f"Loaded {len(results)} document(s)")
    return results


# ── Pure-Python recursive text splitter ───────────────────────────────────────
def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.

    Strategy: try splitting on double newline → single newline → sentence → word.
    Falls back to hard character split only if needed.

    Args:
        text:       Full document text.
        chunk_size: Max characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= chunk_size:
        return [text]

    # Try natural separators in order of preference
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: List[str]) -> List[str]:
        if len(t) <= chunk_size or not seps:
            return [t]

        sep = seps[0]
        parts = t.split(sep)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Part itself too big — recurse with next separator
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: each chunk starts overlap chars before the previous ended
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = overlapped[-1][-overlap:] if len(overlapped[-1]) >= overlap else overlapped[-1]
        overlapped.append(prev_tail + raw_chunks[i])

    return overlapped


# ── Main public function ───────────────────────────────────────────────────────
def chunk_documents(documents: List[tuple[str, str]]) -> List[Chunk]:
    """
    Split all loaded documents into Chunk objects.

    Args:
        documents: Output of load_all_documents() — list of (filename, text).

    Returns:
        Flat list of Chunk objects across all documents.
    """
    all_chunks: List[Chunk] = []

    for source, text in documents:
        raw_chunks = _split_text(text, RAGConfig.chunk_size, RAGConfig.chunk_overlap)
        for i, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():  # skip whitespace-only chunks
                all_chunks.append(Chunk(
                    text=chunk_text.strip(),
                    source=source,
                    chunk_id=i,
                ))

    logger.info(
        f"Chunking complete: {len(all_chunks)} chunks "
        f"(size={RAGConfig.chunk_size}, overlap={RAGConfig.chunk_overlap})"
    )
    return all_chunks


def ingest_documents(docs_dir: str = "data/raw_docs") -> List[Chunk]:
    """
    Full pipeline: load files → split into chunks.

    Args:
        docs_dir: Folder with raw documents.

    Returns:
        List of Chunk objects ready for embedding.
    """
    logger.info(f"Ingesting documents from: {docs_dir}")
    documents = load_all_documents(docs_dir)

    if not documents:
        logger.warning("No documents loaded.")
        return []

    chunks = chunk_documents(documents)
    logger.info(f"Ingestion done: {len(chunks)} chunks ready.")
    return chunks


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_docs"
    chunks = ingest_documents(docs_dir)
    print(f"\n✓ {len(chunks)} chunks created")
    if chunks:
        print(f"\nSample chunk [0]:\n{chunks[0].text[:300]}")
        print(f"Source: {chunks[0].source}, chunk_id: {chunks[0].chunk_id}, size: {chunks[0].chunk_size}")
