# Agentic RAG Assistant

A **production-grade Retrieval-Augmented Generation** system over your PDF
corpus (ships with the Gale Encyclopedia of Medicine). Rebuilt from a basic
Streamlit + HuggingFace prototype into an agentic pipeline on **OpenAI**, served
by **FastAPI** with a **Next.js** chatbot.

Every answer is grounded, cited with numbered footnotes, confidence-scored, and
comes with a full reasoning trace — and it **says when it isn't sure** rather than
bluffing.


```
                       User Query
                            │
                            ▼
                     Router / Planner ──────► (chit-chat? answer directly)
                            │
                            ▼
     Query Rewriter  (rewrite · multi-query · step-back · HyDE · decompose)
                            │
                            ▼
     Hybrid Retrieval  (BM25 + dense vectors, fused with RRF)
                            │
                            ▼
          Cross-Encoder Re-ranker  (top-k)
                            │
                            ▼
        ┌──── Retrieval Verifier ────┐
        │  insufficient?             │ sufficient
   escalate / retry            web-search fallback (CRAG)
        └────────────┬───────────────┘
                     ▼
            Context Compression
                     ▼
                 Generator
                     ▼
     Answer Verifier + Citation / Hallucination check
                     ▼
            Confidence Estimator
                     ▼
        Final Response  (+ persisted trace)
```

## What's implemented

| # | Component | Where |
|---|-----------|-------|
| 1 | **Query rewriter** (context resolution) | `agents/query_rewriter.py` |
| 2 | **Multi-query** expansion | `agents/query_rewriter.py` |
| 3 | **HyDE** (hypothetical document embeddings) | `agents/query_rewriter.py` + `retrieval.py` |
| 4 | **Step-back** prompting | `agents/query_rewriter.py` |
| 5 | **Router / planner** agent | `agents/router.py` |
| 6 | **Decomposition** (multi-hop sub-questions) | `agents/query_rewriter.py` |
| 7 | **Hybrid retrieval** (BM25 + dense) with **RRF** fusion | `retrieval.py`, `vectorstore.py` |
| 8 | **Cross-encoder re-ranker** (LLM fallback) | `rerank.py` |
| 9 | **Retrieval verifier** (CRAG evaluator) | `agents/retrieval_verifier.py` |
| 10 | **Self-healing loop** (retry → escalate → fallback) | `pipeline.py` |
| 11 | **Web-search fallback** (CRAG, optional Tavily) | `agents/web_search.py` |
| 12 | **Context compression** | `compress.py` |
| 13 | **Answer verifier** + **citation / hallucination check** | `agents/answer_verifier.py` |
| 14 | **Confidence estimator** | `agents/confidence.py` |
| 15 | **Retrieval trace / audit log** (persisted) | `trace.py` |

Cost-optimized models: `gpt-4o-mini` for every agent + generation,
`text-embedding-3-small` for embeddings. Every component is individually
toggleable via env vars and **degrades gracefully** — a single agent failure
never breaks the request.

**Honest by design:** passages are cited with clean numbered footnotes (`[1]`,
`[2]`), and if retrieval never verifies as sufficient — even after every
self-healing retry — the answer is capped at **low** confidence and prefixed with
an explicit "I couldn't find strong support for this" note instead of a
confident-but-wrong answer.

## Project layout

```
.
├── backend/                 # FastAPI + the agentic RAG package
│   ├── rag/
│   │   ├── config.py        # all tunables (env-overridable)
│   │   ├── ingest.py        # PDF → chunks → embeddings → hybrid index
│   │   ├── vectorstore.py   # FAISS (dense) + BM25 (sparse)
│   │   ├── retrieval.py     # hybrid retrieval + RRF fusion
│   │   ├── rerank.py        # cross-encoder reranker
│   │   ├── compress.py      # contextual compression
│   │   ├── generate.py      # grounded generation (+ streaming)
│   │   ├── pipeline.py      # the self-healing orchestrator
│   │   ├── trace.py         # audit log
│   │   └── agents/          # router, rewriter, verifiers, web search, confidence
│   ├── app.py               # FastAPI: /health, /chat, /chat/stream
│   ├── requirements.txt
│   └── .env.example
├── frontend/                # Next.js chatbot (streaming, citations, trace UI)
├── data/                    # source PDF(s)
├── docs/DEEP_DIVE.md        # full architecture, rationale, bugs & interview Q&A
├── legacy/                  # the original Streamlit prototype (v0)
└── README.md
```

## Setup

### 1. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
# (optional) set TAVILY_API_KEY=... to enable the web-search fallback
```

Build the index (one-time; embeds the PDF with OpenAI — takes a few minutes and
a few cents for the encyclopedia):

```bash
python -m rag.ingest
```

Run the API:

```bash
uvicorn app:app --reload --port 8000
```

Check it: `curl http://localhost:8000/health`

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # points at http://localhost:8000 by default
npm run dev
```

Open **http://localhost:3000**.

## API

`POST /chat`

```json
{ "message": "What are the symptoms of anemia?", "history": [] }
```

returns

```json
{
  "answer": "Anemia is a deficiency of red blood cells [1]. Symptoms include fatigue [2].",
  "citations": [
    { "id": "1", "label": "Gale...pdf (p.320)", "source": "Gale...pdf", "page": 320, "snippet": "...", "origin": "corpus" },
    { "id": "2", "label": "Gale...pdf (p.321)", "source": "Gale...pdf", "page": 321, "snippet": "...", "origin": "corpus" }
  ],
  "confidence": 0.88,
  "confidence_label": "high",
  "used_web_search": false,
  "heal_attempts": 1,
  "trace_id": "a1b2c3d4e5f6",
  "trace": [ { "name": "router", ... }, ... ]
}
```

Citation `id`s are simple footnote numbers that map 1:1 to the `[n]` markers in
the answer text.

`POST /chat/stream` streams the same via Server-Sent Events
(`meta` → `token`* → `done`). Traces are also written to
`backend/storage/traces/<trace_id>.json`.

## Tuning

Everything in `backend/rag/config.py` is overridable from `.env` — feature
toggles (`ENABLE_*`), retrieval depth (`FINAL_K`, `VECTOR_K`), self-healing
(`MAX_HEAL_ATTEMPTS`, `RETRIEVAL_PASS_THRESHOLD`), and models (`LLM_MODEL`,
`EMBEDDING_MODEL`). To add your own documents, drop PDFs in `data/` and re-run
`python -m rag.ingest`.
