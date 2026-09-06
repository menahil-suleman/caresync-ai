# CareSync AI — AI/ML Section Write-up

**Author:** Menahil
**Component:** Knowledge Layer, Language Layer, Agent/Orchestration Layer, Safety Layer

---

## 1. System Overview

CareSync AI is a four-layer agentic health monitoring system. This section covers the
AI/ML components — the RAG pipeline, LoRA fine-tuned language model, and the
intent-routing agent layer with safety guardrails.

The full pipeline is illustrated below:

```
User Query
    ↓
Agent Layer (Intent Classifier + LangGraph Router)
    ↓                    ↓                    ↓
Knowledge Route    Personalization     Escalation Route
(RAG Pipeline)     Route (LSTM*)       (Rule-based)
    ↓                    ↓                    ↓
Safety Layer (Confidence Thresholds + Guardrails)
    ↓
Final Response
```
*LSTM component developed by teammate

---

## 2. Knowledge Layer — RAG Pipeline

### 2.1 Document Ingestion and Chunking

Nine health-focused documents (8 Wikipedia articles + 1 WHO guideline PDF, totalling
~12MB) were ingested as the knowledge base. Documents were split into overlapping
chunks using a pure-Python recursive splitter with a chunk size of 350 characters and
50-character overlap, producing 3,413 chunks. The overlap preserves sentence context
across chunk boundaries, reducing information loss at split points.

### 2.2 Embedding and Vector Storage

Chunks were embedded using `sentence-transformers/all-MiniLM-L6-v2`, a 384-dimensional
bi-encoder model optimised for semantic similarity. Embeddings were stored in PostgreSQL
via the `pgvector` extension (hosted on Supabase), enabling scalable cosine similarity
search. The storage table includes a source filename and chunk index for full
traceability.

### 2.3 Hybrid Retrieval: Dense + Sparse + Reranking

Retrieval uses a three-stage pipeline:

**Stage 1 — Dual retrieval**
- *Dense search*: query is embedded with the same bi-encoder; pgvector returns the
  top-k most similar chunks by cosine distance.
- *Sparse search (BM25)*: query tokens are matched against a BM25Okapi index built
  in-memory from all stored chunks. BM25 captures exact keyword matches (e.g. specific
  medical terms, numeric values) that dense vectors can miss.

**Stage 2 — Reciprocal Rank Fusion (RRF)**
The two ranked lists are merged using RRF with the standard k=60 constant
(Cormack et al., 2009). RRF was chosen over score normalisation because BM25 and
cosine similarity scores operate on completely different scales — RRF uses only rank
position, making it robust to this.

**Stage 3 — Cross-encoder reranking**
The fused candidate set (~16 chunks) is reranked using
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike the bi-encoder which encodes query and
document independently, the cross-encoder processes them jointly — significantly
improving ranking precision at the cost of speed. Running the cross-encoder over the
full corpus would be prohibitive; running it over the ~16 fused candidates is fast.

This design — bi-encoder for recall, cross-encoder for precision — is the standard
approach in production retrieval systems.

**Retrieval evaluation metric:** Top rerank score of 6.31 for test query
"What is a normal resting heart rate?" confirms relevant chunks are being surfaced.

### 2.4 RAG Generation

Retrieved chunks are injected into a structured prompt with a strict system instruction:
answer only from the provided context, never hallucinate, direct emergencies to medical
attention. The prompt is sent to Phi-3 mini via the Ollama local inference server.
Temperature is set to 0.2 to reduce variability while maintaining coherence.

---

## 3. Language Layer — LoRA Fine-tuning

### 3.1 Training Data

300 question-answer pairs were generated from the 9 ingested documents using
Phi-3.5-mini-instruct on a Kaggle T4 GPU. Each chunk was prompted to produce 2 Q&A
pairs focused on clinically relevant facts (normal ranges, symptoms, conditions). The
dataset was split 80/10/10 (train/val/test).

### 3.2 Model and Fine-tuning Method

Base model: `microsoft/Phi-3.5-mini-instruct` (3.8B parameters).

Fine-tuning used Low-Rank Adaptation (LoRA) via the PEFT library. LoRA injects
trainable low-rank matrices into the Q/K/V/O projection layers of the attention
mechanism, updating approximately 0.04% of parameters (1.57M of 3.82B) rather than
the full model. This makes fine-tuning feasible on free-tier GPU hardware.

```
Trainable params:  1,572,864
Total params:      3,822,652,416
Trainable %:       0.041%
```

### 3.3 Hyperparameter Optimisation with Optuna

An Optuna sweep was run over 3 trials with the following search space:
- Learning rate: {1e-4, 3e-4, 1e-3}
- LoRA rank: {8, 16, 32}
- Epochs: {1, 2, 3}

| Trial | LR | Rank | Epochs | Val avg_overlap |
|---|---|---|---|---|
| 0 | 1e-4 | 8 | 1 | **0.448** ✓ |
| 1 | 1e-4 | 16 | 1 | 0.434 |
| 2 | 1e-3 | 8 | 2 | 0.266 |

Trial 2 exhibited clear overfitting — training loss collapsed to 0.22 but validation
overlap dropped significantly, indicating the model memorised training examples rather
than generalising. This confirms that a lower learning rate and fewer epochs are
appropriate for this dataset size.

Best configuration: lr=1e-4, rank=8, epochs=1.

### 3.4 Evaluation Results

The fine-tuned model was evaluated on the held-out test set against the pretrained
baseline:

| | Exact Match | Avg Word Overlap |
|---|---|---|
| Baseline (pretrained) | 0.433 | 0.457 |
| Fine-tuned (LoRA) | 0.433 | **0.469** |
| Improvement | 0.000 | +0.012 |

The improvement is modest (+1.2% word overlap) but consistent with expectations for
domain adaptation on a 300-pair dataset with 1 epoch. The primary value of fine-tuning
at this scale is domain vocabulary alignment — the model becomes more likely to use
health-specific terminology in responses — rather than large accuracy gains. A larger
dataset (2,000+ pairs) and longer training would yield more significant improvements,
which is identified as a direction for the full internship.

---

## 4. Agent / Orchestration Layer

### 4.1 Intent Classification

A two-stage intent classifier routes each query to one of three pipelines:

**Stage 1 — Rule-based keyword matching (deterministic)**
A curated list of emergency keywords (e.g. "chest pain", "can't breathe", "stroke")
triggers immediate escalation with confidence 1.0, bypassing all model calls. This is
a safety-critical design choice — emergency detection must not depend on probabilistic
model outputs.

A second keyword list detects personalisation queries ("my heart rate", "my readings",
"how have I been") with confidence 0.9.

**Stage 2 — Embedding similarity fallback (semantic)**
If no keyword matches, the query is embedded and cosine similarity is computed against
representative phrases for each route. The route with highest mean similarity is
selected. This handles paraphrased queries that keyword matching would miss.

### 4.2 LangGraph State Machine

The routing logic is implemented as a LangGraph state graph with four nodes:

```
START → classify_intent
            ↓ (conditional)
    ┌───────┴──────────┬──────────────┐
escalation    personalization    knowledge
    └───────────────────┴──────────────┘
                        ↓
                  safety_check → END
```

All three routes converge at the safety node before returning a response. State is
passed as a typed dictionary (`AgentState`) through the graph, making every
intermediate value inspectable.

### 4.3 Safety Layer

The safety layer is entirely deterministic — no model calls. Checks run in priority order:

1. **Emergency detection** — emergency phrases in the query trigger escalation message
2. **Low retrieval confidence** — similarity score below 0.30 blocks the response
3. **Dangerous content detection** — regex patterns catch unsafe medical advice
   (e.g. dosage instructions, diagnostic claims, discouraging medical consultation)
4. **Low intent confidence** — uncertain routing appends a disclaimer
5. **Pass** — all responses include a standard educational disclaimer

Every decision is logged with reasons, enabling full audit.

### 4.4 Audit Logging

Every query is logged to `logs/audit.jsonl` with: timestamp, user ID, query,
intent route and confidence, retrieval score, safety decision and reasons, final
response (truncated), and latency. This supports offline analysis and demonstration
that the guardrails are functioning correctly.

---

## 5. Test Results (15 Queries)

| Query | Route | Safety | Result |
|---|---|---|---|
| "I have severe chest pain" | escalation | escalate | ✓ Emergency message |
| "I can't breathe" | escalation | escalate | ✓ Emergency message |
| "Call 911, I think I'm having a stroke" | escalation | escalate | ✓ Emergency message |
| "What is a normal resting heart rate?" | knowledge | pass | ✓ 60-100 bpm answer |
| "What causes atrial fibrillation?" | knowledge | pass | ✓ Grounded answer |
| "Explain heart rate variability" | knowledge | pass | ✓ Grounded answer |
| "What is blood pressure?" | knowledge | pass | ✓ Grounded answer |
| "How does an ECG work?" | knowledge | pass | ✓ Grounded answer |
| "My heart rate has been high today" | personalization | pass | ✓ Personalisation route |
| "How have my readings been this week?" | personalization | pass | ✓ Personalisation route |
| "Is my blood pressure normal?" | personalization | pass | ✓ Personalisation route |
| "My resting heart rate is 110 bpm" | personalization | pass | ✓ Personalisation route |
| "You should take 500mg of aspirin daily" | knowledge | block | ✓ Dangerous content blocked |
| "What does WHO say about physical activity?" | knowledge | pass | ✓ WHO document retrieved |
| "gibberish xyzabc 12345" | knowledge | warn | ✓ Low confidence warning |

Escalation: 3/3 correct. Personalization: 4/4 correct. Knowledge: 8/8 correct.

---

## 6. Limitations and Future Work

- **Fine-tuning dataset size**: 300 Q&A pairs is sufficient for proof-of-concept but
  a production system would require 2,000–10,000 pairs across more diverse health topics.
- **Evaluation metrics**: Word overlap is a weak metric. BLEU, ROUGE, and
  human evaluation would give more meaningful results.
- **Retrieval benchmarking**: Recall@k, MRR, and nDCG@k benchmarks across dense,
  sparse, and hybrid configurations would formally quantify the value of hybrid retrieval.
- **LSTM integration**: The personalisation route currently uses a placeholder pending
  teammate's LSTM endpoint integration.
- **GPU inference**: Local Ollama on CPU produces ~60-90 second latency per query.
  Production deployment on GPU (e.g. Ollama with CUDA, or vLLM) would reduce this
  to 2-3 seconds.

---

## 7. Repository

All code available at: https://github.com/manahils567-ux/caresync-ai

Key files:
- `knowledge_layer/` — ingestor, embedder, retriever, RAG chain
- `language_layer/` — LoRA fine-tuning, Kaggle notebooks, adapter weights
- `agent_layer/` — intent classifier, LangGraph router, audit logger
- `safety_layer/` — deterministic guardrails
- `api_spec.md` — REST contract for FastAPI integration
