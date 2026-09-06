"""
safety_layer/guardrails.py — Deterministic safety checks outside LLM control

Responsibilities:
  1. Confidence thresholds — flag low-confidence retrievals
  2. Escalation detection — catch emergency patterns missed by intent classifier
  3. Response validation — ensure LLM didn't hallucinate medical advice
  4. Fallback responses — when confidence is too low to answer

This layer is DETERMINISTIC — no model calls, no randomness.
Every decision is rule-based and auditable.

Public API:
    check_response(response: dict) -> SafetyResult
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ── Decision enum ──────────────────────────────────────────────────────────────
class SafetyDecision(str, Enum):
    PASS       = "pass"        # response is safe to return
    WARN       = "warn"        # return with a safety disclaimer appended
    BLOCK      = "block"       # replace with fallback response
    ESCALATE   = "escalate"    # override with emergency message


@dataclass
class SafetyResult:
    decision:        SafetyDecision
    reasons:         List[str] = field(default_factory=list)
    final_response:  str = ""
    disclaimer:      Optional[str] = None


# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_RETRIEVAL_SCORE   = 0.30   # below this → low confidence retrieval
MIN_RERANK_SCORE      = 0.0    # cross-encoder scores can be negative; 0 is threshold
MIN_INTENT_CONFIDENCE = 0.25   # below this → uncertain routing

# ── Canned responses ───────────────────────────────────────────────────────────
ESCALATION_MESSAGE = (
    "⚠️ This sounds like a medical emergency. "
    "Please call emergency services (911) or go to your nearest emergency room immediately. "
    "Do not wait — seek help right now."
)

LOW_CONFIDENCE_FALLBACK = (
    "I don't have enough reliable information to answer that confidently. "
    "Please consult a qualified healthcare professional for accurate advice."
)

SAFETY_DISCLAIMER = (
    "\n\n⚠️ Note: This information is for educational purposes only. "
    "Always consult a healthcare professional before making medical decisions."
)

# ── Patterns that indicate dangerous medical advice ───────────────────────────
DANGEROUS_PATTERNS = [
    r"you should take \d+ mg",
    r"prescribe",
    r"diagnos(e|is|ed)",
    r"you have (cancer|diabetes|heart disease|arrhythmia|afib)",
    r"stop (taking|your) medication",
    r"don't (see|visit|call) (a |the )?doctor",
    r"you don't need (a |the )?doctor",
    r"i (am|'m) (certain|sure) you have",
]

# ── Emergency phrases (second-pass catch) ─────────────────────────────────────
EMERGENCY_PHRASES = [
    "call 911", "emergency room", "go to the hospital immediately",
    "seek immediate", "life-threatening", "cardiac arrest",
]


def _check_dangerous_content(text: str) -> List[str]:
    """Return list of dangerous patterns found in text."""
    found = []
    text_lower = text.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text_lower):
            found.append(pattern)
    return found


def _check_emergency_content(text: str) -> bool:
    """Return True if text contains emergency phrases (needs escalation)."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in EMERGENCY_PHRASES)


def check_response(
    query: str,
    generated_answer: str,
    retrieval_score: Optional[float] = None,
    intent_confidence: Optional[float] = None,
    route: Optional[str] = None,
) -> SafetyResult:
    """
    Run all safety checks on a generated response.

    Checks in priority order:
      1. Emergency detection (always escalate)
      2. Low retrieval confidence (block or warn)
      3. Dangerous medical content (block)
      4. Low intent confidence (warn)
      5. Pass with disclaimer

    Args:
        query:              Original user query.
        generated_answer:   LLM output to validate.
        retrieval_score:    Top retrieval similarity score (0-1).
        intent_confidence:  Intent classifier confidence (0-1).
        route:              Which route was taken (for context).

    Returns:
        SafetyResult with final decision and response text.
    """
    reasons = []

    # ── Check 1: Emergency in query ────────────────────────────────────────────
    query_lower = query.lower()
    emergency_query_patterns = [
        "chest pain", "can't breathe", "heart attack", "stroke",
        "unconscious", "collapsed", "seizure", "dying", "emergency",
    ]
    if any(p in query_lower for p in emergency_query_patterns):
        return SafetyResult(
            decision=SafetyDecision.ESCALATE,
            reasons=["Emergency pattern detected in query"],
            final_response=ESCALATION_MESSAGE,
        )

    # ── Check 2: Emergency content in generated answer ─────────────────────────
    if _check_emergency_content(generated_answer):
        reasons.append("Emergency content detected in response")
        return SafetyResult(
            decision=SafetyDecision.ESCALATE,
            reasons=reasons,
            final_response=ESCALATION_MESSAGE,
        )

    # ── Check 3: Low retrieval confidence ──────────────────────────────────────
    if retrieval_score is not None and retrieval_score < MIN_RETRIEVAL_SCORE:
        reasons.append(
            f"Low retrieval score: {retrieval_score:.3f} < {MIN_RETRIEVAL_SCORE}"
        )
        return SafetyResult(
            decision=SafetyDecision.BLOCK,
            reasons=reasons,
            final_response=LOW_CONFIDENCE_FALLBACK,
        )

    # ── Check 4: Dangerous medical content in answer ───────────────────────────
    dangerous = _check_dangerous_content(generated_answer)
    if dangerous:
        reasons.append(f"Dangerous patterns in response: {dangerous}")
        return SafetyResult(
            decision=SafetyDecision.BLOCK,
            reasons=reasons,
            final_response=LOW_CONFIDENCE_FALLBACK,
        )

    # ── Check 5: Low intent confidence → warn ──────────────────────────────────
    if intent_confidence is not None and intent_confidence < MIN_INTENT_CONFIDENCE:
        reasons.append(
            f"Low intent confidence: {intent_confidence:.3f} < {MIN_INTENT_CONFIDENCE}"
        )
        return SafetyResult(
            decision=SafetyDecision.WARN,
            reasons=reasons,
            final_response=generated_answer + SAFETY_DISCLAIMER,
            disclaimer=SAFETY_DISCLAIMER,
        )

    # ── Check 6: Pass — always add light disclaimer for medical content ─────────
    return SafetyResult(
        decision=SafetyDecision.PASS,
        reasons=["All safety checks passed"],
        final_response=generated_answer + SAFETY_DISCLAIMER,
        disclaimer=SAFETY_DISCLAIMER,
    )


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("I have severe chest pain", "Please seek immediate medical attention.", 0.8, 0.9),
        ("What is normal heart rate?", "Normal resting heart rate is 60-100 bpm.", 0.75, 0.95),
        ("What is normal heart rate?", "I'm certain you have tachycardia.", 0.75, 0.95),
        ("What is normal heart rate?", "I don't know much about this topic.", 0.15, 0.4),
    ]

    print("Safety Layer Test\n" + "="*50)
    for query, answer, ret_score, conf in tests:
        result = check_response(query, answer, ret_score, conf)
        print(f"\nQ: {query[:50]}")
        print(f"   Decision:  {result.decision.value}")
        print(f"   Reasons:   {result.reasons}")
        print(f"   Response:  {result.final_response[:80]}...")
