"""
retriever.py — Cosine similarity search against pgvector (zero LangChain)

Public API:
    retrieve(query, top_k) -> List[dict]
    format_context(chunks) -> str

Each result dict:
    {"content": str, "source": str, "chunk_id": int, "score": float}
"""

from typing import List, Optional

from loguru import logger
from sqlalchemy import text

from config import RAGConfig
from knowledge_layer.embedder import embed_texts, get_engine


def retrieve(query: str, top_k: Optional[int] = None) -> List[dict]:
    """
    Embed query → cosine similarity search → return top-k chunks.

    Args:
        query:  User's question.
        top_k:  How many chunks to return (default: RAGConfig.top_k).

    Returns:
        List of dicts sorted by similarity descending.
    """
    k = top_k or RAGConfig.top_k

    # Embed the query (single string → shape (1, dim) → take [0])
    query_vec = embed_texts([query])[0]

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

    with get_engine().connect() as conn:
        rows = conn.execute(
            sql,
            {"embedding": str(query_vec.tolist()), "k": k}
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

    top = results[0]["score"] if results else "N/A"
    logger.info(f"Retrieved {len(results)} chunks | top score: {top} | query: '{query[:60]}'")
    return results


def format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a single context block for the LLM prompt.

    Args:
        chunks: Output of retrieve().

    Returns:
        String with each chunk numbered, sourced, and scored.
    """
    if not chunks:
        return "No relevant context found."

    parts = [
        f"[{i}] Source: {c['source']} (similarity: {c['score']})\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n---\n\n".join(parts)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is a normal resting heart rate?"
    results = retrieve(q)
    print(f"\nTop {len(results)} results for: '{q}'\n")
    for r in results:
        print(f"  [{r['score']}] {r['source']} — {r['content'][:120]}...")
