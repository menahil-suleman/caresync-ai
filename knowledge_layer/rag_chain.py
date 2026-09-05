"""
rag_chain.py — Full RAG chain: hybrid retrieval → Ollama LLM

Wires together:
  retriever.py  (BM25 + dense + RRF + cross-encoder reranking)
  Ollama        (Phi-3 mini, local inference)

Public API:
    answer(query, bm25_index, user_id) -> dict

Returns:
    {
        "answer":        str,
        "sources":       List[str],
        "chunks":        List[dict],
        "model":         str,
        "retrieval_used": str   # e.g. "hybrid+rerank"
    }
"""

from typing import Optional

import ollama as ollama_client
from loguru import logger

from config import OllamaConfig
from knowledge_layer.retriever import BM25Index, format_context, retrieve


# ── Prompt ─────────────────────────────────────────────────────────────────────
_SYSTEM = """You are CareSync AI, a knowledgeable and careful health information assistant.

Rules you must follow without exception:
1. Answer only using information present in the CONTEXT below.
2. If the context does not contain enough information, say exactly:
   "I don't have enough information to answer that confidently."
3. Never diagnose conditions or recommend treatments.
4. If the question involves urgent symptoms or emergencies, always say:
   "Please seek immediate medical attention."
5. Be concise, accurate, and empathetic."""

_USER_TEMPLATE = """CONTEXT:
{context}

---

QUESTION: {question}

Answer based strictly on the context above:"""


def answer(
    query: str,
    bm25_index: Optional[BM25Index] = None,
    user_id: str = "anonymous",
) -> dict:
    """
    Full RAG pipeline: hybrid retrieve → ground in context → generate.

    Args:
        query:       The user's health question.
        bm25_index:  Pre-built BM25Index for fast sparse retrieval.
                     If None, will be built from DB automatically (slow).
        user_id:     For logging traceability.

    Returns:
        Dict with keys: answer, sources, chunks, model, retrieval_used.
    """
    logger.info(f"[{user_id}] RAG query: '{query[:80]}'")

    # ── Retrieve ───────────────────────────────────────────────────────────────
    chunks = retrieve(query, bm25_index=bm25_index)

    if not chunks:
        logger.warning(f"[{user_id}] No chunks retrieved.")
        return {
            "answer": (
                "I couldn't find relevant information in my knowledge base. "
                "Please consult a healthcare professional."
            ),
            "sources":        [],
            "chunks":         [],
            "model":          OllamaConfig.model,
            "retrieval_used": "hybrid+rerank",
        }

    context = format_context(chunks)
    sources = list({c["source"] for c in chunks})

    # ── Generate ───────────────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            context=context, question=query
        )},
    ]

    try:
        response = ollama_client.chat(
            model=OllamaConfig.model,
            messages=messages,
            options={"temperature": 0.2, "num_predict": 512},
        )
        generated = response["message"]["content"].strip()
        logger.info(f"[{user_id}] Answer generated ({len(generated)} chars).")

    except Exception as e:
        logger.error(f"[{user_id}] Ollama error: {e}")
        generated = (
            "I encountered an error generating a response. "
            "Please try again or consult a healthcare professional."
        )

    return {
        "answer":         generated,
        "sources":        sources,
        "chunks":         chunks,
        "model":          OllamaConfig.model,
        "retrieval_used": "hybrid+rerank",
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from knowledge_layer.retriever import BM25Index

    query = (
        sys.argv[1] if len(sys.argv) > 1
        else "Is it normal for resting heart rate to spike in the evening?"
    )

    print("Building BM25 index from DB...")
    idx = BM25Index.build_from_db()

    result = answer(query, bm25_index=idx)
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"{'='*60}")
    print(f"A: {result['answer']}")
    print(f"\nSources:  {result['sources']}")
    print(f"Model:    {result['model']}")
    print(f"Retrieval: {result['retrieval_used']}")
