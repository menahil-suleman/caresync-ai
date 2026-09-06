"""
intent_classifier.py — Classify user query into one of 3 routes

Routes:
  - ESCALATION      Emergency/urgent symptoms → bypass LLM entirely
  - PERSONALIZATION Query about user's own readings/data → LSTM layer
  - KNOWLEDGE       General health question → RAG pipeline

Two-stage classification:
  1. Rule-based keyword check (fast, deterministic, highest priority)
  2. Embedding similarity fallback (semantic, handles paraphrases)

This is intentionally transparent — every decision is explainable,
which matters for the safety layer audit log.

Public API:
    classify(query: str) -> IntentResult
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger


# ── Route enum ─────────────────────────────────────────────────────────────────
class Route(str, Enum):
    ESCALATION      = "escalation"
    PERSONALIZATION = "personalization"
    KNOWLEDGE       = "knowledge"


@dataclass
class IntentResult:
    route:      Route
    confidence: float        # 0.0 – 1.0
    method:     str          # "keyword" or "embedding"
    matched:    Optional[str] = None   # which keyword/phrase triggered it


# ── Keyword lists ──────────────────────────────────────────────────────────────
# ESCALATION — always hard-routed, no LLM involved
ESCALATION_KEYWORDS = [
    "chest pain", "chest tightness", "heart attack", "cardiac arrest",
    "can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath",
    "stroke", "unconscious", "unresponsive", "collapsed", "collapse",
    "emergency", "call 911", "ambulance", "dying", "severe pain",
    "loss of consciousness", "fainted", "fainting", "seizure",
    "blood pressure spike", "extremely high heart rate", "heart rate over 180",
    "irregular heartbeat severe", "palpitations severe",
]

# PERSONALIZATION — queries about the user's own data/readings
PERSONALIZATION_KEYWORDS = [
    "my heart rate", "my readings", "my data", "my results",
    "my blood pressure", "my pulse", "my ecg", "my ekg",
    "my wearable", "my sensor", "my fitbit", "my apple watch",
    "how have i been", "my trends", "my history", "my stats",
    "today's readings", "yesterday's readings", "this week",
    "am i normal", "is my", "my oxygen", "my spo2",
    "my resting heart rate", "my average",
]

# KNOWLEDGE — everything else defaults here, listed for embedding reference
KNOWLEDGE_PHRASES = [
    "what is", "what are", "explain", "how does", "tell me about",
    "what causes", "symptoms of", "treatment for", "normal range",
    "healthy heart rate", "blood pressure guidelines",
]


def _keyword_match(query: str) -> Optional[IntentResult]:
    """
    Check query against keyword lists.
    Returns IntentResult if matched, None if no match.
    Priority: ESCALATION > PERSONALIZATION > (no match)
    """
    q = query.lower()

    for kw in ESCALATION_KEYWORDS:
        if kw in q:
            return IntentResult(
                route=Route.ESCALATION,
                confidence=1.0,
                method="keyword",
                matched=kw,
            )

    for kw in PERSONALIZATION_KEYWORDS:
        if kw in q:
            return IntentResult(
                route=Route.PERSONALIZATION,
                confidence=0.9,
                method="keyword",
                matched=kw,
            )

    return None


def _embedding_classify(query: str) -> IntentResult:
    """
    Fallback: embed query and compare cosine similarity against
    representative phrases for each route.

    Uses the same embedding model already loaded in embedder.py
    (lazy singleton — no extra memory cost).
    """
    from knowledge_layer.embedder import embed_texts

    # Representative phrases per route
    route_phrases = {
        Route.ESCALATION: [
            "I am having a medical emergency",
            "severe chest pain right now",
            "I think I am having a stroke",
        ],
        Route.PERSONALIZATION: [
            "show me my heart rate data",
            "how have my readings been this week",
            "is my blood pressure normal today",
        ],
        Route.KNOWLEDGE: [
            "what is the normal resting heart rate",
            "explain how ECG works",
            "what causes high blood pressure",
        ],
    }

    # Embed query
    query_vec = embed_texts([query])[0]

    best_route = Route.KNOWLEDGE
    best_score = -1.0

    for route, phrases in route_phrases.items():
        phrase_vecs = embed_texts(phrases)
        # Mean similarity across representative phrases
        sims = [
            float(np.dot(query_vec, pv) / (np.linalg.norm(query_vec) * np.linalg.norm(pv) + 1e-8))
            for pv in phrase_vecs
        ]
        score = float(np.mean(sims))

        if score > best_score:
            best_score = score
            best_route = route

    return IntentResult(
        route=best_route,
        confidence=round(best_score, 4),
        method="embedding",
        matched=None,
    )


def classify(query: str) -> IntentResult:
    """
    Classify a user query into ESCALATION / PERSONALIZATION / KNOWLEDGE.

    Stage 1: keyword match (fast, deterministic)
    Stage 2: embedding similarity (semantic fallback)

    Args:
        query: Raw user input string.

    Returns:
        IntentResult with route, confidence, method, and matched keyword.
    """
    # Stage 1 — keyword
    result = _keyword_match(query)
    if result:
        logger.info(
            f"Intent: {result.route.value} (keyword: '{result.matched}') "
            f"| confidence: {result.confidence}"
        )
        return result

    # Stage 2 — embedding
    result = _embedding_classify(query)
    logger.info(
        f"Intent: {result.route.value} (embedding) "
        f"| confidence: {result.confidence}"
    )
    return result


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "I have severe chest pain and can't breathe",
        "How has my heart rate been today?",
        "What is a normal resting heart rate for adults?",
        "My blood pressure readings have been high this week",
        "What causes atrial fibrillation?",
        "I think I'm having a stroke",
    ]

    print("Intent Classification Test\n" + "="*50)
    for q in test_queries:
        r = classify(q)
        print(f"\nQ: {q}")
        print(f"   → {r.route.value} ({r.method}, conf={r.confidence})"
              + (f" matched='{r.matched}'" if r.matched else ""))
