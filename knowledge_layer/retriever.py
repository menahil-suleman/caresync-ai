"""
retriever.py — Hybrid retrieval: BM25 + dense (pgvector) + cross-encoder reranking

Pipeline:
  1. BM25 sparse search   — keyword overlap, fast, no vectors needed
  2. Dense vector search  — semantic similarity via pgvector cosine
  3. Reciprocal Rank Fusion (RRF) — merge both ranked lists without score scaling issues
  4. Cross-encoder reranking — precision pass: re-scores top candidates with a
                               query-aware model (much more accurate than bi-encoder alone)

This matches "Hybrid Dense + Sparse Retrieval" and "cross-encoder reranking"
exactly as described on the CV — every step is transparent and explainable.

Public API:
    build_bm25_index(chunks)              -> BM25Index (call once after ingestion)
    retrieve(query, bm25_index, top_k)    -> List[dict]
    format_context(results)               -> str

Each result dict:
    {"content": str, "source": str, "chunk_id": int,
     "dense_score": float, "bm25_rank": int, "rerank_score": float}
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sqlalchemy import text

from config import RAGConfig
from knowledge_layer.embedder import embed_texts, get_engine
from knowledge_layer.ingestor import Chunk


# ── BM25 index (built in-memory from stored chunks) ───────────────────────────

@dataclass
class BM25Index:
    """Wraps a BM25Okapi model alongside the original chunk metadata."""
    model:    BM25Okapi
    chunks:   List[dict]   # list of {"content", "source", "chunk_id"}

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "BM25Index":
        """
        Tokenise chunks and build a BM25Okapi index.

        Tokenisation: lowercase + whitespace split (simple but effective for
        health text — no stemming needed at this scale).
        """
        tokenised = [c.text.lower().split() for c in chunks]
        model = BM25Okapi(tokenised)
        meta = [
            {"content": c.text, "source": c.source, "chunk_id": c.chunk_id}
            for c in chunks
        ]
        logger.info(f"BM25 index built over {len(chunks)} chunks.")
        return cls(model=model, chunks=meta)

    @classmethod
    def build_from_db(cls) -> "BM25Index":
        """
        Build BM25 index by reading all chunks from the database.
        Use this when you don't have the original Chunk objects in memory.
        """
        with get_engine().connect() as conn:
            rows = conn.execute(
                text("SELECT content, source, chunk_id FROM document_chunks ORDER BY id")
            ).fetchall()

        if not rows:
            raise RuntimeError("No chunks in DB — run ingest_and_store.py first.")

        chunks = [
            Chunk(text=r.content, source=r.source, chunk_id=r.chunk_id)
            for r in rows
        ]
        return cls.build(chunks)


# ── Cross-encoder (lazy singleton) ────────────────────────────────────────────

_cross_encoder: CrossEncoder | None = None
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder() -> CrossEncoder:
    """Load cross-encoder once and reuse. ~80MB, loads in a few seconds."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder: {_CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder ready.")
    return _cross_encoder


# ── Individual retrieval steps ─────────────────────────────────────────────────

def _dense_search(query: str, k: int) -> List[dict]:
    """
    Semantic search via pgvector cosine similarity.

    Embeds the query with the same bi-encoder used at ingestion time,
    then uses the <=> operator (cosine distance) to find the nearest chunks.
    Returns up to k results with their similarity scores.
    """
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
            sql, {"embedding": str(query_vec.tolist()), "k": k}
        ).fetchall()

    return [
        {
            "content":      row.content,
            "source":       row.source,
            "chunk_id":     row.chunk_id,
            "dense_score":  round(float(row.score), 4),
        }
        for row in rows
    ]


def _bm25_search(query: str, index: BM25Index, k: int) -> List[dict]:
    """
    Sparse BM25 search.

    BM25Okapi scores are term-frequency/IDF-based — captures keyword matches
    that dense vectors sometimes miss (exact medical terms, drug names, values).
    Returns up to k results with their BM25 rank (1 = best).
    """
    tokens = query.lower().split()
    scores = index.model.get_scores(tokens)

    # Get top-k indices by score
    top_indices = np.argsort(scores)[::-1][:k]

    return [
        {
            "content":   index.chunks[i]["content"],
            "source":    index.chunks[i]["source"],
            "chunk_id":  index.chunks[i]["chunk_id"],
            "bm25_rank": rank + 1,   # 1-indexed
            "bm25_score": round(float(scores[i]), 4),
        }
        for rank, i in enumerate(top_indices)
        if scores[i] > 0   # skip zero-score results
    ]


def _reciprocal_rank_fusion(
    dense_results: List[dict],
    bm25_results: List[dict],
    k_rrf: int = 60,
) -> List[dict]:
    """
    Reciprocal Rank Fusion (RRF) merges two ranked lists.

    RRF score = Σ 1 / (k_rrf + rank_i)

    Why RRF instead of score normalisation:
    - BM25 and cosine similarity are on completely different scales
    - RRF only uses *rank position*, making it robust to scale differences
    - k_rrf=60 is the standard empirically-tuned constant (Cormack et al. 2009)

    Returns a merged list sorted by RRF score descending, deduplicated by chunk_id.
    """
    scores: dict[int, float] = {}   # chunk_id -> RRF score
    meta:   dict[int, dict]  = {}   # chunk_id -> result dict

    for rank, result in enumerate(dense_results, 1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
        meta[cid] = result

    for rank, result in enumerate(bm25_results, 1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
        if cid not in meta:
            meta[cid] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids:
        entry = meta[cid].copy()
        entry["rrf_score"] = round(scores[cid], 6)
        merged.append(entry)

    return merged


def _rerank(query: str, candidates: List[dict], top_k: int) -> List[dict]:
    """
    Cross-encoder reranking: precision pass over the merged candidate set.

    The bi-encoder (used in dense search) encodes query and document
    *independently* — fast but less accurate.
    The cross-encoder sees (query, document) *together* — much more accurate
    but too slow to run over the full corpus. We run it only on the ~20
    candidates that survived BM25 + dense fusion.

    Each candidate gets a rerank_score; we return top_k sorted by that score.
    """
    ce = get_cross_encoder()
    pairs = [(query, c["content"]) for c in candidates]
    ce_scores = ce.predict(pairs)   # shape: (len(candidates),)

    for candidate, score in zip(candidates, ce_scores):
        candidate["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ── Main public function ───────────────────────────────────────────────────────

def retrieve(
    query: str,
    bm25_index: Optional[BM25Index] = None,
    top_k: Optional[int] = None,
    use_reranking: bool = True,
) -> List[dict]:
    """
    Full hybrid retrieval pipeline: BM25 + dense → RRF → cross-encoder reranking.

    Args:
        query:         User's health question.
        bm25_index:    Pre-built BM25Index. If None, built from DB automatically.
        top_k:         Final number of chunks to return (default: RAGConfig.top_k).
        use_reranking: Set False to skip cross-encoder (faster, less accurate).

    Returns:
        List of result dicts sorted by rerank_score (or rrf_score if no reranking).
    """
    k = top_k or RAGConfig.top_k

    # Auto-build BM25 index from DB if not provided
    if bm25_index is None:
        logger.warning("No BM25 index provided — building from DB (slow). "
                       "Pass a pre-built BM25Index for production use.")
        bm25_index = BM25Index.build_from_db()

    # Fetch more candidates than needed — reranker will trim to top_k
    fetch_k = min(k * 3, 20)

    # Step 1: dense + sparse
    dense   = _dense_search(query, k=fetch_k)
    sparse  = _bm25_search(query, bm25_index, k=fetch_k)

    logger.debug(f"Dense: {len(dense)} results | BM25: {len(sparse)} results")

    # Step 2: fuse with RRF
    fused = _reciprocal_rank_fusion(dense, sparse)
    logger.debug(f"After RRF fusion: {len(fused)} unique candidates")

    # Step 3: cross-encoder reranking
    if use_reranking and fused:
        results = _rerank(query, fused, top_k=k)
        sort_key = "rerank_score"
    else:
        results = fused[:k]
        sort_key = "rrf_score"

    top_score = results[0].get(sort_key, "N/A") if results else "N/A"
    logger.info(
        f"Retrieved {len(results)} chunks | top {sort_key}: {top_score} "
        f"| query: '{query[:60]}'"
    )
    return results


def format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.

    Shows source, similarity/rerank score, and chunk text.
    """
    if not chunks:
        return "No relevant context found."

    score_key = "rerank_score" if "rerank_score" in chunks[0] else "rrf_score"
    parts = [
        f"[{i}] Source: {c['source']} ({score_key}: {c.get(score_key, 'N/A')})\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n---\n\n".join(parts)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is a normal resting heart rate?"

    print(f"Query: '{q}'\n")
    print("Building BM25 index from DB...")
    idx = BM25Index.build_from_db()
    results = retrieve(q, bm25_index=idx)

    print(f"\nTop {len(results)} results:\n")
    for r in results:
        print(f"  [rerank: {r.get('rerank_score','N/A')} | rrf: {r.get('rrf_score','N/A')}]")
        print(f"  {r['source']} — {r['content'][:120]}...\n")
