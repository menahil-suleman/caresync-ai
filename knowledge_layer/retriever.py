"""
retriever.py — Semantic retrieval from pgvector

Given a user query, this module:
  1. Embeds the query using the same model used during ingestion
  2. Runs cosine similarity search against document_chunks in pgvector
  3. Returns top-k chunks as plain strings (ready for LLM context)

Public API:
    retrieve(query: str, top_k: int = None) -> List[dict]

Each returned dict has:
    {
        "content": str,       # chunk text
        "source":  str,       # filename it came from
        "chunk_id": int,      # position within source doc
        "score":   float      # cosine similarity (0–1, higher = more relevant)
    }
"""

from typing import List, Optional

from loguru import logger
from sqlalchemy import text

from config import DBConfig, RAGConfig
from knowledge_layer.embedder import embed_texts, get_engine


def retrieve(query: str, top_k: Optional[int] = None) -> List[dict]:
    """
    Semantic search: embed query → cosine similarity → top-k chunks.

    Args:
        query:  User question or search string.
        top_k:  Number of chunks to return. Defaults to RAGConfig.top_k.

    Returns:
        List of dicts with keys: content, source, chunk_id, score.
        Sorted by similarity descending (best match first).
    """
    k = top_k or RAGConfig.top_k

    # 1. Embed the query
    query_embedding = embed_texts([query])[0]  # shape: (dim,)

    # 2. Query pgvector using <=> (cosine distance operator)
    #    cosine similarity = 1 - cosine_distance
    sql = text("""
        SELECT
            content,
            source,
            chunk_id,
            1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM document_chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :k
    """)

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                "embedding": str(query_embedding.tolist()),
                "k": k,
            },
        ).fetchall()

    results = [
        {
            "content":  row.content,
            "source":   row.source,
            "chunk_id": row.chunk_id,
            "score":    round(float(row.score), 4),
        }
        for row in rows
    ]

    logger.info(
        f"Retrieved {len(results)} chunks for query: '{query[:60]}...' "
        f"(top score: {results[0]['score'] if results else 'N/A'})"
    )
    return results


def format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Each chunk is prefixed with its source filename and similarity score.

    Args:
        chunks: Output of retrieve().

    Returns:
        Multi-line string to inject as [CONTEXT] in the LLM prompt.
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['source']} (similarity: {chunk['score']})\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is a normal resting heart rate?"
    chunks = retrieve(query)
    print(f"\n✓ Top-{len(chunks)} results for: '{query}'\n")
    for c in chunks:
        print(f"  [{c['score']}] {c['source']} — {c['content'][:120]}...")
