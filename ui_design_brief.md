# CareSync AI — UI/UX Design Brief

---

## What is CareSync AI?

CareSync AI is a privacy-preserving personal health monitoring assistant. It answers
health questions using a knowledge base of medical documents, detects emergencies and
immediately escalates them, and is designed to eventually connect to wearable sensor
data for personalised health insights.

The system is built around a core principle: **AI assists, never diagnoses.**
Every response is grounded in verified medical sources, and dangerous queries are
intercepted before they ever reach the language model.

---

## Who uses it?

**Primary user:** A health-conscious individual (20–45 years old) who:
- Wears a fitness tracker or smartwatch
- Wants to understand their health data without Googling
- Asks questions like "Is my resting heart rate normal?" or "What causes HRV to drop?"
- May occasionally ask something urgent — the system must handle that safely

**Secondary user:** A researcher or developer reviewing the system's audit logs and
routing decisions.

---

## Application Type

Single-page web app / Streamlit prototype.
Think: a calm, clinical chat interface — not a flashy consumer app.
The tone is **trustworthy, clean, medical-adjacent** — similar to how a health app
from Apple or a digital health startup would look.

---

## Core Screens

### 1. Chat Interface (Main Screen)

This is where the user spends 95% of their time.

**Layout:**
- Left sidebar (~25% width): branding, navigation, stats
- Main area (~75% width): chat messages + input

**Left Sidebar contains:**
- CareSync AI logo + tagline ("Your AI Health Assistant")
- Navigation links: Chat, My Health Data (placeholder), About
- Small stats panel at bottom showing:
  - Total queries today
  - Route breakdown (Knowledge / Personalization / Escalation)
  - Average response confidence

**Chat area:**
- Clean message bubbles
  - User messages: right-aligned, dark/accent colour
  - AI messages: left-aligned, white/light grey with subtle border
- Each AI message card shows:
  - The response text
  - A small metadata row below the text (subtle, not intrusive):
    - Route badge: `🧠 Knowledge` / `👤 Personalisation` / `⚠️ Escalation`
    - Confidence: e.g. `conf: 0.87`
    - Sources: e.g. `📄 Heart rate - Wikipedia.pdf`
    - Latency: e.g. `1.2s`
- Emergency responses look different:
  - Red/orange banner background
  - Large ⚠️ icon
  - Bold text
  - "Call 911" or "Seek emergency care" clearly visible

**Input area (bottom of chat):**
- Text input field: "Ask a health question..."
- Send button
- Small disclaimer below input:
  "CareSync AI provides educational information only. Not a substitute for medical advice."

**Suggested questions (shown when chat is empty):**
- "What is a normal resting heart rate?"
- "What causes atrial fibrillation?"
- "How does heart rate variability relate to stress?"
- "What does an ECG measure?"

---

### 2. Route Explanation Panel (Expandable)

When user clicks on a route badge (e.g. `🧠 Knowledge`), a small expandable panel
appears below the message explaining how the answer was generated:

```
🧠 Knowledge Route
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retrieval method:  Hybrid (BM25 + dense + cross-encoder reranking)
Sources used:      Heart rate - Wikipedia.pdf, Electrocardiography - Wikipedia.pdf
Retrieval score:   6.31 (cross-encoder)
Safety check:      ✅ Passed
```

This is for the demo — shows reviewers the system is working correctly under the hood.

---

### 3. About / System Info Page

Simple static page explaining the system:

**Sections:**
- What is CareSync AI? (2-3 sentences)
- How it works (simple 4-step diagram):
  1. You ask a question
  2. AI classifies your intent (Knowledge / Personalisation / Escalation)
  3. Relevant documents are retrieved and ranked
  4. Answer is generated and safety-checked before returning to you
- Tech stack (clean icon grid):
  - Phi-3.5-mini (Language Model)
  - pgvector + Supabase (Vector Database)
  - BM25 + Dense + Cross-encoder (Retrieval)
  - LangGraph (Agent Routing)
  - LoRA / PEFT (Fine-tuning)
- Disclaimer box

---

## Visual Design Direction

**Colour palette:**
- Primary background: `#0F1117` (dark, like a medical monitor)
- Secondary background: `#1A1D26` (slightly lighter for cards)
- Accent / brand colour: `#4F9CF9` (calm medical blue)
- Emergency colour: `#FF4B4B` (red for escalation messages)
- Warning colour: `#FFA500` (orange for low-confidence warnings)
- Text primary: `#FAFAFA`
- Text secondary: `#8B8FA8`
- Success green: `#00C9A7`

**Typography:**
- Font: Inter or SF Pro (clean, readable, medical-adjacent)
- Message text: 14-15px
- Headers: 18-20px medium weight
- Metadata: 11-12px secondary colour

**Route badge colours:**
- 🧠 Knowledge: blue pill `#4F9CF9`
- 👤 Personalisation: purple pill `#A78BFA`
- ⚠️ Escalation: red pill `#FF4B4B`
- Safety pass: green dot `#00C9A7`
- Safety warn: orange dot `#FFA500`
- Safety block: red dot `#FF4B4B`

---

## Component Breakdown

### Message Card (AI Response)
```
┌─────────────────────────────────────────────────────┐
│ 🤖  CareSync AI                                      │
│                                                       │
│  A normal resting heart rate for adults is between   │
│  60 and 100 beats per minute (bpm). Athletes may     │
│  have lower resting rates of 40–60 bpm.              │
│                                                       │
│  ⚠️ Note: This is educational information only.      │
│  Always consult a healthcare professional.            │
│                                                       │
│  ────────────────────────────────────────────────   │
│  🧠 Knowledge  •  conf: 0.87  •  📄 Heart rate.pdf  │
│  1.2s                                       [expand] │
└─────────────────────────────────────────────────────┘
```

### Emergency Card (Escalation Response)
```
┌─────────────────────────────────────────────────────┐
│ ⚠️  MEDICAL EMERGENCY DETECTED                      │  ← red background
│                                                       │
│  This sounds like a medical emergency.               │
│  Please call emergency services (911) or go to       │
│  your nearest emergency room immediately.            │
│  Do not wait — seek help right now.                  │
│                                                       │
│  ⚠️ Escalation  •  conf: 1.0  •  Rule-based         │
└─────────────────────────────────────────────────────┘
```

### Sidebar Stats Panel
```
┌─────────────────────┐
│  Today's Activity   │
│  ─────────────────  │
│  Queries:      12   │
│  Knowledge:     8   │
│  Personal:      3   │
│  Escalations:   1   │
│  ─────────────────  │
│  Avg latency: 1.2s  │
│  Avg conf:   0.84   │
└─────────────────────┘
```

---

## What the UI does NOT do

- No user authentication (prototype)
- No actual wearable data connection (personalization shows placeholder)
- No medical records storage
- No multi-turn conversation memory (each query is independent)

---

## Data the UI receives per query

When a user sends a message, the backend returns this JSON:

```json
{
  "answer": "A normal resting heart rate is 60-100 bpm...",
  "route": "knowledge",
  "safety_decision": "pass",
  "sources": ["Heart rate - Wikipedia.pdf"],
  "intent_confidence": 0.87,
  "latency_ms": 1243.5
}
```

The UI uses all these fields to render the message card.

---

## Suggested Screens to Design

1. **Chat (empty state)** — with suggested questions, clean header
2. **Chat (active)** — 3-4 messages showing different routes
3. **Chat (emergency state)** — red escalation card prominent
4. **Expanded route panel** — showing retrieval details
5. **About page** — system diagram + tech stack

---

## Tone and Feel

- **Not**: flashy, gamified, consumer-app energy
- **Yes**: calm, clinical, trustworthy, precise
- Reference apps: Apple Health, Headspace (minimalist), Linear (dark + clean)
- The user should feel like they're talking to a knowledgeable but careful assistant,
  not a chatbot

---

## Notes for the Developer (after Stitch designs it)

- Built in Streamlit
- Backend is `handle_query(query, user_id)` from `agent_layer/router.py`
- No FastAPI needed — call the Python function directly
- BM25 index should be built once at app startup and cached in `st.session_state`
- Audit log at `logs/audit.jsonl` can be read for the stats panel
- Emergency responses should use `st.error()` or a custom red container
