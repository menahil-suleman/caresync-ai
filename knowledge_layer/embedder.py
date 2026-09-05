"""
embedder.py — Embedding + pgvector storage (zero LangChain)

Responsibilities:
  1. Load sentence-transformer embedding model (singleton)
  2. Create pgvector table if not exists
  3. Embed Chunk objects and store in PostgreSQL

Public API:
    store_chunks(chunks: List[Chunk]) -> None
    embed_texts(texts: List[str])     -> np.ndarray
"""

from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from config import DBConfig, EmbeddingConfig
from knowledge_layer.ingestor import Chunk


# ── Lazy singleton: embedding model ───────────────────────────────────────────
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EmbeddingConfig.model_name}")
        _embedding_model = SentenceTransformer(EmbeddingConfig.model_name)
        logger.info("Embedding model ready.")
    return _embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns:
        numpy array shape (N, embedding_dim)
    """
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=True, batch_size=32)


# ── Database ───────────────────────────────────────────────────────────────────
def get_engine():
    return create_engine(DBConfig.dsn(), pool_pre_ping=True)


def init_db(engine) -> int:
    """
    Create pgvector extension + document_chunks table + index if missing.

    Returns:
        Embedding dimension.
    """
    dim = get_embedding_model().get_sentence_embedding_dimension()

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id        SERIAL PRIMARY KEY,
                source    TEXT,
                chunk_id  INTEGER,
                content   TEXT NOT NULL,
                embedding vector({dim})
            );
        """))
        # IVFFlat index for fast approximate nearest-neighbour search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))
        conn.commit()

    logger.info(f"DB ready — table 'document_chunks' (dim={dim})")
    return dim


def store_chunks(chunks: List[Chunk]) -> None:
    """
    Embed all chunks and insert into pgvector, skipping duplicates.

    Args:
        chunks: Output of ingestor.ingest_documents()
    """
    if not chunks:
        logger.warning("store_chunks: nothing to store.")
        return

    engine = get_engine()
    init_db(engine)

    texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    inserted = skipped = 0

    with Session(engine) as session:
        for chunk, embedding in zip(chunks, embeddings):
            # Skip if (source, chunk_id) already in DB — safe to re-run
            exists = session.execute(
                text(
                    "SELECT 1 FROM document_chunks "
                    "WHERE source = :src AND chunk_id = :cid LIMIT 1"
                ),
                {"src": chunk.source, "cid": chunk.chunk_id},
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
                    "source":    chunk.source,
                    "chunk_id":  chunk.chunk_id,
                    "content":   chunk.text,
                    "embedding": str(embedding.tolist()),
                },
            )
            inserted += 1

        session.commit()

    logger.info(f"Storage done — inserted: {inserted}, skipped: {skipped}")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = ["Heart rate variability reflects autonomic nervous system health.",
              "Resting heart rate above 100 bpm is classified as tachycardia."]
    vecs = embed_texts(sample)
    print(f"✓ Shape: {vecs.shape}  (should be (2, 384) for all-MiniLM-L6-v2)")
