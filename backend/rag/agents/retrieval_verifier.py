"""Retrieval verifier (CRAG-style evaluator).

After retrieval + rerank, judge whether the chunks actually contain enough
information to answer the question. Returns a 0-1 score and what's missing.
The self-healing loop uses this to decide whether to retry / escalate / fall
back to web search.
"""
from __future__ import annotations

from typing import List

from ..config import get_settings
from ..llm import parse
from ..schemas import Chunk, RetrievalVerdict


def _format_docs(docs: List[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {d.text[:600]}" for i, d in enumerate(docs))


def verify(query: str, docs: List[Chunk]) -> RetrievalVerdict:
    settings = get_settings()
    if not docs:
        return RetrievalVerdict(
            sufficient=False, score=0.0, missing_information="no documents retrieved"
        )
    if not settings.enable_retrieval_verifier:
        return RetrievalVerdict(sufficient=True, score=1.0)

    system = (
        "You judge whether retrieved passages are sufficient to fully and "
        "accurately answer the user's question. Consider coverage and specificity. "
        "score: 0.0 (useless) to 1.0 (fully sufficient). List concrete missing "
        "information if any."
    )
    user = f"Question: {query}\n\nRetrieved passages:\n{_format_docs(docs)}"
    try:
        verdict = parse(system, user, RetrievalVerdict, max_tokens=400)
    except Exception:
        # Fail open: don't block answering on a verifier hiccup.
        return RetrievalVerdict(sufficient=True, score=0.6, reasoning="verifier-fallback")
    verdict.sufficient = verdict.score >= settings.retrieval_pass_threshold
    return verdict
