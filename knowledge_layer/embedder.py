"""
embedder.py — Embedding + pgvector storage module

Responsibilities:
  1. Load sentence-transformer embedding model
  2. Create pgvector table (if not exists)
  3. Embed chunks and store them in PostgreSQL via pgvector
  4. Expose a simple store_chunks() function

Usage:
    from knowledge_layer.embedder import store_chunks
    store_chunks(chunks)
"""

from typing import List

import numpy as np
from langchain.schema import Document
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, Integer, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column

from config import DBConfig, EmbeddingConfig

# ── pgvector SQLAlchemy setup ──────────────────────────────────────────────────
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logger.warning("pgvector not installed — store_chunks will be a no-op.")


# ── Lazy singleton: embedding model ───────────────────────────────────────────
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse across calls."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EmbeddingConfig.model_name}")
        _embedding_model = SentenceTransformer(EmbeddingConfig.model_name)
        logger.info("Embedding model loaded.")
    return _embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Args:
        texts: Plain strings to embed.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings  # shape: (N, dim)


# ── Database helpers ───────────────────────────────────────────────────────────

def get_engine():
    return create_engine(DBConfig.dsn(), pool_pre_ping=True)


def init_db(engine) -> int:
    """
    Create the pgvector extension and the document_chunks table if missing.

    Returns embedding dimension so callers can verify.
    """
    # Probe embedding dim
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                source      TEXT,
                chunk_id    INTEGER,
                content     TEXT NOT NULL,
                embedding   vector({dim})
            );
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))
        conn.commit()

    logger.info(f"DB initialised — table 'document_chunks' ready (dim={dim}).")
    return dim


# ── Main public function ───────────────────────────────────────────────────────

def store_chunks(chunks: List[Document]) -> None:
    """
    Embed all chunks and upsert them into the pgvector table.

    Skips chunks whose (source, chunk_id) pair already exists to allow
    re-runs without duplicating data.

    Args:
        chunks: Output of knowledge_layer.ingestor.ingest_documents().
    """
    if not PGVECTOR_AVAILABLE:
        logger.error("pgvector unavailable — cannot store chunks.")
        return

    if not chunks:
        logger.warning("store_chunks called with empty list.")
        return

    engine = get_engine()
    init_db(engine)

    texts = [c.page_content for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    inserted = 0
    skipped = 0

    with Session(engine) as session:
        for chunk, embedding in zip(chunks, embeddings):
            source = chunk.metadata.get("source", "unknown")
            chunk_id = chunk.metadata.get("chunk_id", -1)

            # Check for existing record
            exists = session.execute(
                text(
                    "SELECT 1 FROM document_chunks "
                    "WHERE source = :src AND chunk_id = :cid LIMIT 1"
                ),
                {"src": source, "cid": chunk_id},
            ).fetchone()

            if exists:
                skipped += 1
                continue

            session.execute(
                text(
                    "INSERT INTO document_chunks (source, chunk_id, content, embedding) "
                    "VALUES (:source, :chunk_id, :content, :embedding)"
                ),
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "content": chunk.page_content,
                    "embedding": embedding.tolist(),
                },
            )
            inserted += 1

        session.commit()

    logger.info(
        f"Storage complete — inserted: {inserted}, skipped (already exist): {skipped}"
    )


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "Heart rate variability is a key indicator of autonomic nervous system balance.",
        "Resting heart rate above 100 bpm is classified as tachycardia.",
    ]
    embeddings = embed_texts(sample_texts)
    print(f"✓ Embedded {len(embeddings)} texts, shape: {embeddings.shape}")
