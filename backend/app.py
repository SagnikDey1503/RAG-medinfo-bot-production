"""FastAPI service exposing the agentic RAG pipeline.

Endpoints:
  GET  /health          -> liveness + index status
  POST /chat            -> blocking answer (full trace, citations, confidence)
  POST /chat/stream     -> Server-Sent Events: meta -> token* -> done

The pipeline (and its FAISS/BM25 index) is loaded once at startup and reused.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.config import get_settings
from rag.pipeline import RAGPipeline
from rag.schemas import ChatRequest, ChatResponse
from rag.vectorstore import HybridStore

_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()
    if settings.openai_api_key and HybridStore.exists():
        try:
            _pipeline = RAGPipeline()
            print(f"[startup] pipeline ready ({_pipeline.store.size} chunks)")
        except Exception as e:  # pragma: no cover
            print(f"[startup] failed to load pipeline: {e}")
    else:
        print("[startup] pipeline NOT loaded (missing OPENAI_API_KEY or index). "
              "Set the key and run `python -m rag.ingest`.")
    yield


app = FastAPI(title="Agentic RAG API", version="1.0.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline unavailable. Ensure OPENAI_API_KEY is set and the index "
            "is built (`python -m rag.ingest`), then restart the server.",
        )
    return _pipeline


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "index_exists": HybridStore.exists(),
        "chunks": _pipeline.store.size if _pipeline else 0,
        "web_search_enabled": _settings.web_search_enabled,
        "model": _settings.llm_model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    return _require_pipeline().answer(req)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    pipeline = _require_pipeline()

    def event_gen():
        try:
            for event_type, payload in pipeline.stream_answer(req):
                if event_type == "done":
                    payload = payload.model_dump()
                elif event_type == "meta":
                    pass  # already a dict
                data = payload if isinstance(payload, (dict,)) else {"text": payload}
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
