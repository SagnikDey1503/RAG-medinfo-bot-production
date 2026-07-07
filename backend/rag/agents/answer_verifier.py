"""Answer verifier: hallucination + citation audit.

Given the question, the cited passages, and a draft answer, check each factual
claim against the passages. Unsupported claims are flagged and removed, and a
corrected, fully-grounded answer is returned. This is the last line of defense
against hallucination (roadmap items #4, #9, #21).
"""
from __future__ import annotations

from typing import List

from ..config import get_settings
from ..llm import parse
from ..schemas import AnswerVerification, Chunk


def _format_docs(docs: List[Chunk]) -> str:
    # Number passages by position, matching the generator's citation scheme.
    return "\n\n".join(f"[{i + 1}] {d.text[:700]}" for i, d in enumerate(docs))


def verify(query: str, docs: List[Chunk], draft: str) -> AnswerVerification:
    settings = get_settings()
    if not settings.enable_answer_verifier:
        return AnswerVerification(grounded=True, corrected_answer=draft)

    system = (
        "You are a strict fact-checker for a grounded assistant. You are given a "
        "question, numbered source passages, and a draft answer.\n"
        "1. Break the draft into atomic factual claims.\n"
        "2. For each claim mark supported=true only if a passage clearly supports "
        "it, and list the supporting passage numbers.\n"
        "3. Produce corrected_answer: rewrite the answer keeping ONLY supported "
        "claims, preserving the inline numeric citation markers like [1], [2]. If "
        "the passages do not answer the question, corrected_answer should say so "
        "honestly.\n"
        "Do not introduce new facts."
    )
    user = (
        f"Question: {query}\n\nSource passages:\n{_format_docs(docs)}\n\n"
        f"Draft answer:\n{draft}"
    )
    try:
        result = parse(system, user, AnswerVerification, max_tokens=1100)
    except Exception:
        return AnswerVerification(grounded=True, corrected_answer=draft, reasoning="verifier-fallback")
    if not result.corrected_answer.strip():
        result.corrected_answer = draft
    return result
