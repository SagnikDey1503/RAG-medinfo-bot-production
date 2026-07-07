"""Confidence estimator.

Combines signals already computed by the pipeline into a single 0-1 score and a
human label. No extra LLM call — cheap and deterministic.

Signals:
  * retrieval verifier score (evidence quality)
  * fraction of answer claims that were supported (grounding)
  * whether the answer verifier had to strip unsupported content
  * whether we fell back to web search (slightly lower confidence)
"""
from __future__ import annotations

from typing import Tuple

from ..schemas import AnswerVerification, RetrievalVerdict


def _label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def estimate(
    retrieval: RetrievalVerdict,
    answer: AnswerVerification,
    *,
    used_web: bool,
    heal_attempts: int,
) -> Tuple[float, str]:
    claims = answer.claims or []
    grounding = (
        sum(1 for c in claims if c.supported) / len(claims) if claims else (1.0 if answer.grounded else 0.5)
    )

    score = 0.5 * retrieval.score + 0.5 * grounding
    if answer.unsupported_removed:
        score -= 0.1
    if used_web:
        score -= 0.05
    if heal_attempts > 1:
        score -= 0.05 * (heal_attempts - 1)

    # The verifier never confirmed the evidence was sufficient even after every
    # self-healing retry (and web fallback, if enabled). Claim-level grounding
    # only checks "is this sentence backed by *some* retrieved chunk" — it can't
    # tell if that chunk is actually about the right topic. So when retrieval
    # itself never passed, cap confidence low regardless of how well-grounded
    # the answer looks, rather than let a topically-wrong-but-textually-
    # consistent answer read as trustworthy.
    if not retrieval.sufficient:
        score = min(score, 0.35)

    score = max(0.0, min(1.0, score))
    return round(score, 3), _label(score)
