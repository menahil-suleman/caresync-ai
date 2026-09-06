# CareSync AI — REST API Spec
> For teammate to implement in FastAPI

---

## POST /query

Main entry point. Routes query through agent layer.

**Request:**
```json
{
  "user_id": "user_123",
  "query": "What is a normal resting heart rate?"
}
```

**Response:**
```json
{
  "answer": "A normal resting heart rate for adults is 60–100 bpm...",
  "route": "knowledge",
  "safety_decision": "pass",
  "sources": ["Heart rate - Wikipedia.pdf"],
  "intent_confidence": 0.87,
  "latency_ms": 1243.5
}
```

---

## POST /personalization/check

Called by personalization_node in agent layer.
**Teammate implements this.**

**Request:**
```json
{
  "user_id": "user_123",
  "query": "How has my heart rate been today?"
}
```

**Response:**
```json
{
  "flagged": false,
  "confidence": 0.82,
  "contributing_features": ["heart_rate", "hrv"],
  "message": "Your heart rate has been within normal range today (avg 72 bpm).",
  "window_start": "2026-09-06T00:00:00Z",
  "window_end": "2026-09-06T23:59:59Z"
}
```

---

## GET /logs

Returns recent audit log entries.

**Response:**
```json
{
  "total": 42,
  "entries": [
    {
      "timestamp": "2026-09-06T12:00:00Z",
      "user_id": "user_123",
      "query": "...",
      "intent_route": "knowledge",
      "safety_decision": "pass",
      "latency_ms": 1243.5
    }
  ]
}
```

---

## GET /stats

Summary stats from audit log.

**Response:**
```json
{
  "total": 100,
  "routes": {"knowledge": 70, "personalization": 20, "escalation": 10},
  "safety_decisions": {"pass": 85, "warn": 10, "block": 4, "escalate": 1},
  "avg_latency_ms": 1150.3
}
```

---

## Notes for teammate
- Agent layer (`agent_layer/router.py`) calls `handle_query(query, user_id)` — single entry point
- Personalization node currently has a placeholder — wire in your LSTM endpoint at `POST /personalization/check`
- All responses go through safety layer before returning — don't bypass it
- Audit logs are in `logs/audit.jsonl`
