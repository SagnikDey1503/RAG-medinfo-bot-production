"""Central configuration for the RAG service.

All tunables live here and are overridable via environment variables (or a
`.env` file). Keeping them in one place makes the system easy to operate in
production without touching code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (backend/rag/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Provider keys -----------------------------------------------------
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    # Optional web-search fallback (CRAG). If unset, web search is disabled.
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

    # ---- Models (cost-optimized tier) --------------------------------------
    # Small, cheap model used for every agent call AND final generation.
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    # A slightly stronger fallback used only when self-healing escalates.
    llm_model_strong: str = Field(default="gpt-4o-mini", alias="LLM_MODEL_STRONG")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    # ---- Paths -------------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    index_dir: Path = Field(default=REPO_ROOT / "backend" / "storage" / "index", alias="INDEX_DIR")
    trace_dir: Path = Field(default=REPO_ROOT / "backend" / "storage" / "traces", alias="TRACE_DIR")

    # ---- Chunking ----------------------------------------------------------
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    embed_batch_size: int = Field(default=256, alias="EMBED_BATCH_SIZE")

    # ---- Retrieval ---------------------------------------------------------
    # How many candidates each retriever pulls before fusion.
    vector_k: int = Field(default=20, alias="VECTOR_K")
    bm25_k: int = Field(default=20, alias="BM25_K")
    # Reciprocal-rank-fusion constant.
    rrf_k: int = Field(default=60, alias="RRF_K")
    # Candidates handed to the reranker.
    fusion_k: int = Field(default=24, alias="FUSION_K")
    # Final docs kept after reranking and fed to the generator.
    final_k: int = Field(default=6, alias="FINAL_K")

    # ---- Agent behaviour ---------------------------------------------------
    enable_query_rewrite: bool = Field(default=True, alias="ENABLE_QUERY_REWRITE")
    enable_multi_query: bool = Field(default=True, alias="ENABLE_MULTI_QUERY")
    enable_hyde: bool = Field(default=True, alias="ENABLE_HYDE")
    enable_router: bool = Field(default=True, alias="ENABLE_ROUTER")
    enable_decomposition: bool = Field(default=True, alias="ENABLE_DECOMPOSITION")
    enable_rerank: bool = Field(default=True, alias="ENABLE_RERANK")
    enable_retrieval_verifier: bool = Field(default=True, alias="ENABLE_RETRIEVAL_VERIFIER")
    enable_answer_verifier: bool = Field(default=True, alias="ENABLE_ANSWER_VERIFIER")
    enable_compression: bool = Field(default=True, alias="ENABLE_COMPRESSION")
    # Skip compressing chunks already shorter than this (not worth an LLM call).
    compression_min_chars: int = Field(default=350, alias="COMPRESSION_MIN_CHARS")
    enable_web_fallback: bool = Field(default=True, alias="ENABLE_WEB_FALLBACK")

    # Self-healing loop: how many retrieve/verify cycles before giving up.
    max_heal_attempts: int = Field(default=3, alias="MAX_HEAL_ATTEMPTS")
    # Retrieval verifier must score >= this (0-1) to be considered sufficient.
    retrieval_pass_threshold: float = Field(default=0.6, alias="RETRIEVAL_PASS_THRESHOLD")
    # Number of multi-query variants to generate.
    multi_query_count: int = Field(default=4, alias="MULTI_QUERY_COUNT")

    # Cross-encoder model (local, free). Empty => fall back to LLM reranker.
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="CROSS_ENCODER_MODEL"
    )

    # ---- Server ------------------------------------------------------------
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")
    request_timeout_s: int = Field(default=90, alias="REQUEST_TIMEOUT_S")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.enable_web_fallback and self.tavily_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
