"""Query rewriting / expansion agent.

Combines several well-known techniques in one structured call:
  * Rewrite      — resolve pronouns/context into a standalone question.
  * Multi-query  — N diverse paraphrases to widen recall.
  * Step-back    — a more general question that surfaces background context.
  * HyDE         — a hypothetical answer paragraph, embedded for dense recall.
  * Decompose    — sub-questions for multi-hop queries.

The `aggressive` flag (used when the self-healing loop escalates) asks for more
variants and a longer HyDE document.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import get_settings
from ..llm import parse
from ..schemas import ChatMessage, QueryExpansion


def _history_block(history: Optional[List[ChatMessage]]) -> str:
    if not history:
        return "(no prior conversation)"
    turns = history[-6:]
    return "\n".join(f"{m.role}: {m.content}" for m in turns)


def expand(
    query: str,
    *,
    history: Optional[List[ChatMessage]] = None,
    multi_hop: bool = False,
    aggressive: bool = False,
) -> QueryExpansion:
    settings = get_settings()
    n = settings.multi_query_count + (2 if aggressive else 0)

    want = []
    if settings.enable_query_rewrite:
        want.append("a single standalone 'rewritten' question (resolve context)")
    if settings.enable_multi_query:
        want.append(f"{n} diverse 'multi_queries' paraphrases using varied terminology")
    want.append("a 'step_back' more general question")
    if settings.enable_hyde:
        length = "2-4 sentence" if aggressive else "1-2 sentence"
        want.append(f"a {length} 'hyde_document': a hypothetical ideal answer passage")
    if multi_hop and settings.enable_decomposition:
        want.append("'sub_queries': the atomic sub-questions needed to answer")

    system = (
        "You rewrite and expand search queries for a retrieval system. Produce:\n- "
        + "\n- ".join(want)
        + "\nBe faithful to the user's intent; never invent facts."
    )
    user = (
        f"Conversation so far:\n{_history_block(history)}\n\n"
        f"Current user question: {query}"
    )
    try:
        result = parse(system, user, QueryExpansion, max_tokens=700)
    except Exception:
        return QueryExpansion(rewritten=query, multi_queries=[], step_back=None)

    if not result.rewritten.strip():
        result.rewritten = query
    return result


def all_query_variants(query: str, expansion: QueryExpansion) -> List[str]:
    """Flatten the expansion into a de-duplicated list of literal query strings."""
    variants = [query, expansion.rewritten]
    variants += expansion.multi_queries
    variants += expansion.sub_queries
    if expansion.step_back:
        variants.append(expansion.step_back)
    seen, out = set(), []
    for v in variants:
        v = (v or "").strip()
        key = v.lower()
        if v and key not in seen:
            seen.add(key)
            out.append(v)
    return out
