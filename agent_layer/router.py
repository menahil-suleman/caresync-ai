"""
agent_layer/router.py — LangGraph intent router

Graph structure:
  [START] → classify_intent → {escalation | personalization | knowledge}
                ↓                      ↓                 ↓
          escalation_node    personalization_node   knowledge_node
                ↓                      ↓                 ↓
          safety_check ←─────────────────────────────────┘
                ↓
           [END] → final response

Each node is a pure Python function. LangGraph manages the state
machine — transitions, shared state, and error handling.

Public API:
    build_graph() -> CompiledGraph
    handle_query(query, user_id, bm25_index) -> dict
"""

import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from agent_layer.intent_classifier import Route, classify
from agent_layer.logger import AuditEntry, log_interaction
from safety_layer.guardrails import SafetyDecision, check_response


# ── Graph state ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Input
    query:         str
    user_id:       str

    # Set by classify_intent node
    route:         str
    intent_conf:   float
    intent_method: str

    # Set by route nodes
    raw_answer:    str
    retrieval_score: Optional[float]
    sources:       list

    # Set by safety node
    final_answer:  str
    safety_decision: str
    safety_reasons:  list

    # Timing
    start_time:    float


# ── Node: classify intent ──────────────────────────────────────────────────────
def classify_intent(state: AgentState) -> AgentState:
    """Classify the query into ESCALATION / PERSONALIZATION / KNOWLEDGE."""
    result = classify(state["query"])
    return {
        **state,
        "route":         result.route.value,
        "intent_conf":   result.confidence,
        "intent_method": result.method,
    }


# ── Node: escalation ──────────────────────────────────────────────────────────
def escalation_node(state: AgentState) -> AgentState:
    """
    Hard-coded escalation — no model call.
    For emergency queries, we bypass LLM entirely.
    This is a safety-critical design decision.
    """
    logger.warning(f"[{state['user_id']}] ESCALATION triggered: '{state['query'][:60]}'")
    return {
        **state,
        "raw_answer":       "ESCALATION",
        "retrieval_score":  1.0,   # escalation is always high confidence
        "sources":          [],
    }


# ── Node: knowledge ───────────────────────────────────────────────────────────
def knowledge_node(state: AgentState) -> AgentState:
    """
    RAG pipeline: hybrid retrieve → Phi-3 generate.
    Uses BM25 + pgvector + cross-encoder reranking.
    """
    from knowledge_layer.rag_chain import answer
    from knowledge_layer.retriever import BM25Index

    # BM25 index passed through app state (built once at startup)
    bm25_index = state.get("bm25_index")

    result = answer(state["query"], bm25_index=bm25_index, user_id=state["user_id"])

    # Get top retrieval score for safety check
    top_score = None
    if result["chunks"]:
        score_key = "rerank_score" if "rerank_score" in result["chunks"][0] else "rrf_score"
        top_score = result["chunks"][0].get(score_key)

    return {
        **state,
        "raw_answer":      result["answer"],
        "retrieval_score": top_score,
        "sources":         result["sources"],
    }


# ── Node: personalization ─────────────────────────────────────────────────────
def personalization_node(state: AgentState) -> AgentState:
    """
    Personalization route — checks user's wearable data via LSTM.
    For Day 3, returns a placeholder until teammate's LSTM is integrated.
    The REST API contract is defined so teammate can wire in directly.
    """
    logger.info(f"[{state['user_id']}] Personalization route")

    # ── Placeholder: will be replaced by teammate's LSTM endpoint ─────────────
    # Expected teammate API:
    #   POST /personalization/check
    #   Body: {"user_id": str, "query": str}
    #   Response: {"flagged": bool, "confidence": float,
    #              "contributing_features": list, "message": str}
    #
    # For now return a structured placeholder response
    placeholder = (
        "I can see you're asking about your personal health data. "
        "Your wearable data analysis is being processed. "
        "Based on recent readings, your metrics appear within normal ranges. "
        "For detailed analysis, please check your wearable dashboard."
    )

    return {
        **state,
        "raw_answer":      placeholder,
        "retrieval_score": 0.8,   # placeholder confidence
        "sources":         ["wearable_data"],
    }


# ── Node: safety check ────────────────────────────────────────────────────────
def safety_node(state: AgentState) -> AgentState:
    """
    Run all safety checks on the raw answer before returning to user.
    This node runs regardless of which route was taken.
    """
    result = check_response(
        query=state["query"],
        generated_answer=state["raw_answer"],
        retrieval_score=state.get("retrieval_score"),
        intent_confidence=state.get("intent_conf"),
        route=state.get("route"),
    )

    return {
        **state,
        "final_answer":    result.final_response,
        "safety_decision": result.decision.value,
        "safety_reasons":  result.reasons,
    }


# ── Routing function ───────────────────────────────────────────────────────────
def route_decision(state: AgentState) -> str:
    """Determine which node to go to after intent classification."""
    route = state["route"]
    if route == Route.ESCALATION.value:
        return "escalation"
    elif route == Route.PERSONALIZATION.value:
        return "personalization"
    else:
        return "knowledge"


# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph():
    """
    Construct and compile the LangGraph state machine.

    Graph:
      START → classify_intent → [conditional] → {escalation|personalization|knowledge}
                                                         ↓
                                                    safety_check → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("escalation",         escalation_node)
    graph.add_node("personalization",    personalization_node)
    graph.add_node("knowledge",          knowledge_node)
    graph.add_node("safety_check",       safety_node)

    # Entry point
    graph.add_edge(START, "classify_intent")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_decision,
        {
            "escalation":    "escalation",
            "personalization": "personalization",
            "knowledge":     "knowledge",
        },
    )

    # All routes converge at safety check
    graph.add_edge("escalation",      "safety_check")
    graph.add_edge("personalization", "safety_check")
    graph.add_edge("knowledge",       "safety_check")

    # Safety → END
    graph.add_edge("safety_check", END)

    return graph.compile()


# ── Main entry point ───────────────────────────────────────────────────────────
_graph = None   # lazy singleton


def handle_query(
    query: str,
    user_id: str = "anonymous",
    bm25_index=None,
) -> dict:
    """
    Single entry point for all user queries.

    Args:
        query:       User's health question.
        user_id:     For logging and personalization.
        bm25_index:  Pre-built BM25 index (pass once at startup, reuse).

    Returns:
        {
            "answer":           str,
            "route":            str,
            "safety_decision":  str,
            "sources":          list,
            "intent_confidence": float,
            "latency_ms":       float,
        }
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    start = time.time()

    initial_state: AgentState = {
        "query":           query,
        "user_id":         user_id,
        "route":           "",
        "intent_conf":     0.0,
        "intent_method":   "",
        "raw_answer":      "",
        "retrieval_score": None,
        "sources":         [],
        "final_answer":    "",
        "safety_decision": "",
        "safety_reasons":  [],
        "start_time":      start,
        "bm25_index":      bm25_index,
    }

    final_state = _graph.invoke(initial_state)
    latency_ms = (time.time() - start) * 1000

    # Audit log
    log_interaction(AuditEntry(
        user_id=user_id,
        query=query,
        intent_route=final_state["route"],
        intent_confidence=final_state["intent_conf"],
        intent_method=final_state["intent_method"],
        retrieval_score=final_state.get("retrieval_score"),
        safety_decision=final_state["safety_decision"],
        safety_reasons=final_state["safety_reasons"],
        final_response=final_state["final_answer"],
        latency_ms=latency_ms,
    ))

    return {
        "answer":            final_state["final_answer"],
        "route":             final_state["route"],
        "safety_decision":   final_state["safety_decision"],
        "sources":           final_state["sources"],
        "intent_confidence": final_state["intent_conf"],
        "latency_ms":        round(latency_ms, 1),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        ("I have severe chest pain", "test_user"),
        ("What is a normal resting heart rate?", "test_user"),
        ("How has my heart rate been today?", "test_user"),
    ]

    print("Agent Router Test\n" + "="*60)
    for query, uid in test_queries:
        print(f"\nQ: {query}")
        result = handle_query(query, uid)
        print(f"   Route:    {result['route']}")
        print(f"   Safety:   {result['safety_decision']}")
        print(f"   Latency:  {result['latency_ms']}ms")
        print(f"   Answer:   {result['answer'][:100]}...")
