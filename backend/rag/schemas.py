"""Pydantic models shared across the pipeline and the HTTP API.

Two families live here:
  * Internal dataclass-style models used to pass data between pipeline stages.
  * API request/response models exposed by FastAPI.
"""
from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Retrieval primitives                                                        #
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    """A single retrievable unit of text plus provenance."""

    id: str
    text: str
    source: str = "unknown"
    page: Optional[int] = None
    # Populated as the chunk flows through the pipeline.
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    origin: Literal["corpus", "web"] = "corpus"

    def citation_label(self) -> str:
        page = f"p.{self.page}" if self.page is not None else "?"
        return f"{self.source} ({page})"


# --------------------------------------------------------------------------- #
# Agent outputs (structured LLM responses)                                    #
# --------------------------------------------------------------------------- #
class RouteDecision(BaseModel):
    """Router agent: where should this query go and how complex is it."""

    needs_retrieval: bool = Field(
        description="False for pure chit-chat/meta questions that need no documents."
    )
    is_multi_hop: bool = Field(
        description="True if answering requires chaining several facts / sub-questions."
    )
    domain: str = Field(default="general", description="Best-guess topical domain.")
    reasoning: str = ""


class QueryExpansion(BaseModel):
    """Query-rewriter output: rewrites, variants, step-back, and HyDE doc."""

    rewritten: str = Field(description="Single best standalone rewrite of the query.")
    sub_queries: List[str] = Field(default_factory=list)
    multi_queries: List[str] = Field(default_factory=list)
    step_back: Optional[str] = None
    hyde_document: Optional[str] = None


class RetrievalVerdict(BaseModel):
    """Retrieval verifier: are the retrieved docs good enough to answer?"""

    sufficient: bool
    score: float = Field(ge=0.0, le=1.0)
    missing_information: str = ""
    reasoning: str = ""


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    citation_ids: List[str] = Field(default_factory=list)


class AnswerVerification(BaseModel):
    """Answer verifier: hallucination + citation audit, with a corrected answer."""

    grounded: bool
    claims: List[ClaimCheck] = Field(default_factory=list)
    corrected_answer: str
    unsupported_removed: bool = False
    reasoning: str = ""


# --------------------------------------------------------------------------- #
# API surface                                                                 #
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    # Per-request feature overrides (optional). None = use server default.
    stream: bool = False


class Citation(BaseModel):
    id: str
    label: str
    source: str
    page: Optional[int] = None
    snippet: str
    origin: Literal["corpus", "web"] = "corpus"


class TraceStep(BaseModel):
    name: str
    detail: str = ""
    duration_ms: Optional[float] = None
    data: Optional[dict] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: str = "unknown"
    used_web_search: bool = False
    heal_attempts: int = 1
    trace_id: str
    trace: List[TraceStep] = Field(default_factory=list)
