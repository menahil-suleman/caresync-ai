"""
agent_layer/logger.py — Audit logger for every query and routing decision

Logs every interaction to:
  1. logs/audit.jsonl   — structured JSONL for analysis
  2. loguru console     — human-readable

Each log entry contains:
  - timestamp, user_id, query
  - intent route + confidence + method
  - retrieval scores
  - safety decision
  - final response (truncated)
  - latency_ms

Public API:
    log_interaction(entry: AuditEntry) -> None
    load_logs(n: int) -> List[dict]
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger

AUDIT_LOG_FILE = Path("logs/audit.jsonl")


@dataclass
class AuditEntry:
    user_id:            str
    query:              str
    intent_route:       str
    intent_confidence:  float
    intent_method:      str
    retrieval_score:    Optional[float]
    safety_decision:    str
    safety_reasons:     List[str]
    final_response:     str          # first 200 chars only
    latency_ms:         float
    timestamp:          str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    model_used:         str = "phi3:mini"


def log_interaction(entry: AuditEntry) -> None:
    """
    Write audit entry to JSONL file and loguru console.

    Args:
        entry: AuditEntry dataclass with all interaction details.
    """
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = asdict(entry)
    # Truncate response for log
    record["final_response"] = entry.final_response[:200]

    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        f"[{entry.user_id}] {entry.intent_route} | "
        f"safety={entry.safety_decision} | "
        f"conf={entry.intent_confidence:.2f} | "
        f"{entry.latency_ms:.0f}ms"
    )


def load_logs(n: int = 100) -> List[dict]:
    """Load last n audit log entries."""
    if not AUDIT_LOG_FILE.exists():
        return []
    lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    return [json.loads(l) for l in lines[-n:]]


def get_stats() -> dict:
    """Basic stats from audit log — useful for demo."""
    logs = load_logs(1000)
    if not logs:
        return {"total": 0}

    routes = {}
    safety = {}
    for log in logs:
        r = log.get("intent_route", "unknown")
        s = log.get("safety_decision", "unknown")
        routes[r] = routes.get(r, 0) + 1
        safety[s] = safety.get(s, 0) + 1

    return {
        "total":           len(logs),
        "routes":          routes,
        "safety_decisions": safety,
        "avg_latency_ms":  round(sum(l.get("latency_ms", 0) for l in logs) / len(logs), 1),
    }
