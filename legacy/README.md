# Legacy (v0) — Streamlit + HuggingFace prototype

These are the **original** files, kept for reference. They are superseded by the
production system in [`../backend`](../backend) and [`../frontend`](../frontend).

- `one.py`  — built a 384-dim FAISS index with HuggingFace MiniLM embeddings
- `two.py`  — CLI `RetrievalQA`
- `medbot.py` — Streamlit chat UI (`k=3`, Mistral-7B via HuggingFace Inference)
- `requirements.txt` — old dependency set

The new system replaces all of this with an agentic OpenAI RAG pipeline
(query rewriting, hybrid retrieval, reranking, self-healing, verified citations)
served by FastAPI with a Next.js chatbot. See the top-level `README.md`.
