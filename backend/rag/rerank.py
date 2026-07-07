"""Cross-encoder reranker.

Preferred path: a local sentence-transformers CrossEncoder (free at inference,
genuine query-document cross-attention). If torch / the model is unavailable,
we degrade gracefully to a cheap LLM listwise reranker so the pipeline still
works on minimal installs.
"""
from __future__ import annotations

from typing import List, Optional

from .config import get_settings
from .schemas import Chunk

_cross_encoder = None
_cross_encoder_failed = False


def _get_cross_encoder():
    """Lazily load the cross-encoder; cache success/failure."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is not None or _cross_encoder_failed:
        return _cross_encoder
    settings = get_settings()
    if not settings.cross_encoder_model:
        _cross_encoder_failed = True
        return None
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    except Exception:
        _cross_encoder_failed = True
        _cross_encoder = None
    return _cross_encoder


def _llm_rerank(query: str, docs: List[Chunk], top_k: int) -> List[Chunk]:
    """Fallback: ask the LLM to score relevance 0-10 for each doc."""
    from .llm import parse
    from pydantic import BaseModel

    class _Score(BaseModel):
        index: int
        score: float

    class _Scores(BaseModel):
        scores: List[_Score]

    listing = "\n\n".join(
        f"[{i}] {d.text[:500]}" for i, d in enumerate(docs)
    )
    try:
        result = parse(
            system=(
                "You are a search relevance judge. Score how well each passage "
                "answers the user question from 0 (irrelevant) to 10 (perfect). "
                "Return a score for every passage index."
            ),
            user=f"Question: {query}\n\nPassages:\n{listing}",
            schema=_Scores,
            max_tokens=1200,
        )
        score_map = {s.index: s.score for s in result.scores}
    except Exception:
        return docs[:top_k]
    for i, d in enumerate(docs):
        d.rerank_score = score_map.get(i, 0.0)
    return sorted(docs, key=lambda d: d.rerank_score or 0.0, reverse=True)[:top_k]


def rerank(query: str, docs: List[Chunk], top_k: Optional[int] = None) -> List[Chunk]:
    settings = get_settings()
    top_k = top_k or settings.final_k
    if not docs:
        return []

    model = _get_cross_encoder()
    if model is None:
        return _llm_rerank(query, docs, top_k)

    pairs = [(query, d.text) for d in docs]
    scores = model.predict(pairs)
    for d, s in zip(docs, scores):
        d.rerank_score = float(s)
    return sorted(docs, key=lambda d: d.rerank_score or 0.0, reverse=True)[:top_k]
