"""
rag_chain.py — Wire retrieval → Ollama LLM into a complete RAG chain

Public API:
    answer(query: str, user_id: str = "anonymous") -> dict

Returns:
    {
        "answer":   str,          # LLM-generated answer
        "sources":  List[str],    # filenames used
        "chunks":   List[dict],   # raw retrieved chunks (for logging/debug)
        "model":    str           # model name used
    }
"""

from typing import Optional

import ollama as ollama_client
from loguru import logger

from config import OllamaConfig, RAGConfig
from knowledge_layer.retriever import format_context, retrieve

# ── Prompt template ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CareSync AI, a knowledgeable and cautious health information assistant.

Rules you MUST follow:
1. Only answer based on the provided CONTEXT. Do not hallucinate.
2. If the context does not contain enough information, say:
   "I don't have enough information to answer that confidently."
3. Never diagnose medical conditions or prescribe treatments.
4. For emergencies or urgent symptoms, always direct the user to seek medical attention.
5. Be concise, clear, and empathetic.
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

---

USER QUESTION: {question}

Answer based strictly on the context above:"""


def build_prompt(query: str, context: str) -> list[dict]:
    """Build the messages list for Ollama chat API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=context, question=query
            ),
        },
    ]


def answer(query: str, user_id: str = "anonymous") -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks → generate grounded answer.

    Args:
        query:   The user's health question.
        user_id: For logging/traceability.

    Returns:
        Dict with keys: answer, sources, chunks, model.
    """
    logger.info(f"[{user_id}] RAG query: '{query[:80]}'")

    # ── Step 1: Retrieve ───────────────────────────────────────────────────────
    chunks = retrieve(query)

    if not chunks:
        logger.warning(f"[{user_id}] No chunks retrieved — returning fallback.")
        return {
            "answer": (
                "I couldn't find relevant information in my knowledge base. "
                "Please consult a healthcare professional for accurate advice."
            ),
            "sources": [],
            "chunks": [],
            "model": OllamaConfig.model,
        }

    context = format_context(chunks)
    sources = list({c["source"] for c in chunks})

    # ── Step 2: Generate ───────────────────────────────────────────────────────
    messages = build_prompt(query, context)

    try:
        response = ollama_client.chat(
            model=OllamaConfig.model,
            messages=messages,
            options={"temperature": 0.2, "num_predict": 512},
        )
        generated_answer = response["message"]["content"].strip()
        logger.info(f"[{user_id}] Answer generated ({len(generated_answer)} chars).")

    except Exception as e:
        logger.error(f"[{user_id}] Ollama error: {e}")
        generated_answer = (
            "I encountered an error generating a response. "
            "Please try again or consult a healthcare professional."
        )

    return {
        "answer":  generated_answer,
        "sources": sources,
        "chunks":  chunks,
        "model":   OllamaConfig.model,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Is it normal for resting heart rate to spike in the evening?"
    )

    result = answer(query)
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"{'='*60}")
    print(f"A: {result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Model:   {result['model']}")
