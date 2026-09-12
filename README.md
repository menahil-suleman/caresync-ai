# CareSync AI

**Privacy-preserving personal health monitoring powered by a four-layer agentic AI system.**

> Built from primitives — no LangChain, no pre-built RAG frameworks.
> Every retrieval, routing, and safety decision is transparent and explainable.

---

## What it does

CareSync AI answers health questions grounded in verified medical sources, detects
emergency queries and escalates them without involving a language model, and is
architected to connect to wearable sensor data for personalised anomaly detection.

A user asks a question. Within seconds:

1. A two-stage intent classifier determines whether this is a general health question,
   a personal data query, or a medical emergency
2. Emergency queries are intercepted by rule-based logic and returned with an
   escalation message — the language model is never called
3. General queries go through a hybrid retrieval pipeline (BM25 + dense vectors +
   cross-encoder reranking) to surface the most relevant passages from a curated
   medical knowledge base
4. The retrieved context and query are sent to a LoRA fine-tuned Phi-3.5-mini model
   for grounded answer generation
5. Every response passes through a deterministic safety layer before reaching the user
6. Every interaction is logged to a structured audit file

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│         Agent / Orchestration Layer      │
│  ┌─────────────────────────────────┐    │
│  │  Intent Classifier              │    │
│  │  Stage 1: Keyword rules         │    │
│  │  Stage 2: Embedding similarity  │    │
│  └──────────┬──────────────────────┘    │
│             │ LangGraph State Machine    │
│    ┌────────┼────────┐                  │
│    ▼        ▼        ▼                  │
│  Escal.  Person.  Knowledge             │
└────│────────│────────│──────────────────┘
     │        │        │
     │        │        ▼
     │        │  ┌─────────────────────────┐
     │        │  │   Knowledge Layer       │
     │        │  │  BM25 sparse search     │
     │        │  │  + pgvector dense search│
     │        │  │  + RRF fusion           │
     │        │  │  + Cross-encoder rerank │
     │        │  │  → Phi-3.5-mini (LoRA)  │
     │        │  └─────────────────────────┘
     │        │
     │        ▼
     │  ┌─────────────────────────┐
     │  │  Personalization Layer  │
     │  │  Wearable data analysis │
     │  │  (LSTM anomaly detect.) │
     │  └─────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│            Safety Layer                  │
│  Confidence thresholds                   │
│  Dangerous content detection             │
│  Emergency escalation                    │
│  Audit logging                           │
└─────────────────────────────────────────┘
     │
     ▼
 Response
```

---

## Key Numbers

| Metric | Value |
|---|---|
| Documents ingested | 9 (8 Wikipedia + 1 WHO guideline) |
| Knowledge base size | ~12 MB |
| Chunks stored in pgvector | 3,413 |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| Retrieval pipeline stages | 3 (BM25 + dense + cross-encoder) |
| Q&A pairs for fine-tuning | 300 |
| Base model | Phi-3.5-mini-instruct (3.8B params) |
| Trainable parameters (LoRA) | 1,572,864 (0.041% of model) |
| Optuna trials | 3 |
| Baseline avg word overlap | 0.457 |
| Fine-tuned avg word overlap | 0.469 (+1.2%) |
| Intent classification accuracy | 15/15 test queries correct |
| Escalation route latency | ~25ms (rule-based, no model call) |
| Knowledge route latency (CPU) | ~60-90s (GPU: ~2-3s) |

---

## Retrieval Pipeline

Standard dense-only retrieval has two failure modes: it misses exact keyword matches
(e.g. specific drug names, numeric thresholds), and bi-encoder similarity scores
are not well-calibrated for ranking. This system addresses both.

**BM25 (sparse)** captures term-frequency matches that semantic vectors miss.
**Dense search via pgvector** captures semantic similarity for paraphrased queries.
**Reciprocal Rank Fusion (k=60)** merges the two ranked lists by position, not score —
making it robust to the scale difference between BM25 and cosine similarity.
**Cross-encoder reranking** runs a query-aware model over the ~16 fused candidates.
The cross-encoder sees query and document jointly (unlike the bi-encoder which encodes
them independently), producing significantly more accurate relevance scores.
The precision gain is achieved without the latency cost of running the cross-encoder
over the full corpus.

---

## Fine-tuning

Phi-3.5-mini-instruct was fine-tuned using Low-Rank Adaptation (LoRA) on 300
domain-specific Q&A pairs generated from the ingested documents. LoRA injects
trainable low-rank matrices into the attention projection layers, updating 0.041%
of parameters — making fine-tuning feasible on free-tier GPU hardware (Kaggle T4).

Hyperparameters were selected via an Optuna sweep:

| Trial | LR | LoRA Rank | Epochs | Val Overlap |
|---|---|---|---|---|
| 0 | 1e-4 | 8 | 1 | **0.448** ✓ best |
| 1 | 1e-4 | 16 | 1 | 0.434 |
| 2 | 1e-3 | 8 | 2 | 0.266 (overfit) |

Trial 2 demonstrates clear overfitting — training loss collapsed to 0.22 while
validation overlap dropped to 0.266, confirming that higher learning rates are
unsuitable for this dataset size.

---

## Intent Routing

The intent classifier uses a two-stage approach designed for reliability in a
safety-critical context:

**Stage 1 — Rule-based keyword matching**
A curated set of emergency phrases (e.g. "chest pain", "can't breathe", "cardiac arrest")
triggers immediate escalation with confidence 1.0. This is intentionally deterministic —
emergency detection must not depend on probabilistic model outputs. A second keyword
list detects personalisation queries. Both stages execute in < 1ms.

**Stage 2 — Embedding similarity fallback**
For queries that match no keywords, the query is embedded and cosine similarity is
computed against representative phrases for each route. This handles paraphrased
queries keyword matching would miss.

The LangGraph state machine connects three route nodes (escalation, personalization,
knowledge) all converging at a safety node before returning to the user.

---

## Safety Layer

Every response — regardless of route — passes through a deterministic safety layer
before reaching the user. Checks run in priority order:

1. Emergency phrase detection in query → escalation message, no model involved
2. Low retrieval confidence (score < 0.30) → blocked, fallback response returned
3. Dangerous content detection via regex (dosage instructions, diagnostic claims,
   discouraging medical consultation) → blocked
4. Low intent confidence (< 0.25) → response returned with disclaimer
5. Pass → response returned with standard educational disclaimer

Every decision is logged with reasons to `logs/audit.jsonl`.

---

## Project Structure

```
caresync_ai/
│
├── knowledge_layer/
│   ├── ingestor.py          # PDF/text loading + pure-Python chunking
│   ├── embedder.py          # sentence-transformers + pgvector storage
│   ├── retriever.py         # BM25 + dense + RRF + cross-encoder reranking
│   └── rag_chain.py         # retrieve → generate via Ollama
│
├── language_layer/
│   ├── finetune_phi3.py     # LoRA fine-tuning pipeline (local)
│   ├── kaggle_notebook.py   # Full training script for Kaggle T4 GPU
│   ├── kaggle_qa_gen.py     # Q&A pair generation on Kaggle GPU
│   └── phi3_caresync_lora/  # Saved LoRA adapter + evaluation results
│
├── agent_layer/
│   ├── intent_classifier.py # Two-stage: keyword rules + embedding similarity
│   ├── router.py            # LangGraph state machine + handle_query() entry point
│   └── logger.py            # Structured audit logging to JSONL
│
├── safety_layer/
│   └── guardrails.py        # Deterministic safety checks (no model calls)
│
├── data/
│   ├── raw_docs/            # Source documents (9 health PDFs)
│   └── qa_pairs/            # 300 generated Q&A pairs (qa_pairs.jsonl)
│
├── logs/                    # Audit logs (audit.jsonl)
├── config.py                # Centralised configuration from .env
├── ingest_and_store.py      # One-shot ingestion pipeline runner
├── api_spec.md              # REST API contract
└── writeup_ai_ml_section.md # Technical write-up
```

---

## Setup

**Prerequisites:** Python 3.12, Ollama, Supabase account (free tier)

```bash
# 1. Clone
git clone https://github.com/manahils567-ux/caresync-ai.git
cd caresync-ai

# 2. Virtual environment
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
# Edit .env — add your Supabase DATABASE_URL and set OLLAMA_MODEL=phi3:mini

# 5. Pull the language model
ollama pull phi3:mini

# 6. Enable pgvector on Supabase
# In Supabase SQL Editor: CREATE EXTENSION IF NOT EXISTS vector;

# 7. Ingest documents
# Drop PDF/text files into data/raw_docs/
python ingest_and_store.py

# 8. Test the full pipeline
python -m agent_layer.router
```

---

## Running a Query

```python
from agent_layer.router import handle_query
from knowledge_layer.retriever import BM25Index

# Build BM25 index once at startup
bm25_index = BM25Index.build_from_db()

# Query the system
result = handle_query(
    query="What is a normal resting heart rate?",
    user_id="user_001",
    bm25_index=bm25_index,
)

print(result["answer"])
# → "A normal resting heart rate for adults is between 60 and 100 bpm..."
print(result["route"])          # → "knowledge"
print(result["safety_decision"]) # → "pass"
print(result["intent_confidence"]) # → 0.87
```

---

## Design Decisions

**Why not LangChain?**
The CV lists pgvector, hybrid retrieval, cross-encoder reranking, and agentic routing
as skills built from scratch. LangChain abstracts all of these into pre-made classes.
Building from primitives means every component is explainable in a research interview:
the chunking logic, the RRF fusion formula, the cross-encoder reranking step, and the
safety check order are all visible in the codebase with no hidden framework behaviour.

**Why LangGraph for routing?**
The routing logic is a state machine with conditional transitions — exactly what
LangGraph is designed for. The graph structure makes the routing logic explicit and
visualisable, unlike a chain of if/else statements.

**Why deterministic safety checks?**
A language model cannot reliably detect its own dangerous outputs. The safety layer
uses regex and threshold rules that are predictable, auditable, and cannot be
prompt-injected. Emergency escalation in particular must never depend on a
probabilistic model.

**Why LoRA over full fine-tuning?**
Full fine-tuning of a 3.8B parameter model requires 40+ GB VRAM and hours of compute.
LoRA updates 0.041% of parameters, fits on a free T4 GPU (15GB), and trains in under
3 minutes per trial — making iterative Optuna sweeps feasible within a 4-day build.

---

## Evaluation Summary

| Component | Metric | Result |
|---|---|---|
| Retrieval | Top rerank score (test query) | 6.31 |
| Fine-tuning | Baseline word overlap | 0.457 |
| Fine-tuning | Fine-tuned word overlap | 0.469 (+1.2%) |
| Intent routing | Accuracy (15 test queries) | 15/15 (100%) |
| Safety layer | Emergency queries caught | 3/3 (100%) |
| Safety layer | Dangerous content blocked | 1/1 (100%) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language model | Phi-3.5-mini-instruct (Microsoft) |
| Fine-tuning | LoRA via PEFT / Unsloth |
| Hyperparameter search | Optuna |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Vector database | pgvector (PostgreSQL via Supabase) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Agent routing | LangGraph |
| Local LLM inference | Ollama |
| ORM | SQLAlchemy |
| Logging | Loguru |

---

## Acknowledgements

Knowledge base sources: Wikipedia (CC BY-SA), WHO (CC BY-NC-SA 3.0 IGO).
This system is a research prototype. It does not provide medical advice.

---

*Built by Menahil*
