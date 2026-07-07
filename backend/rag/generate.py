"""Answer generation.

Builds the grounded prompt and produces an answer with inline numbered citation
markers ([1], [2], ...). Passages are numbered by position, so the model never
sees (and can never leak) internal chunk ids, and the streamed text shows clean
footnote numbers immediately. Exposes both a blocking and a streaming API; both
share one prompt builder so behaviour is identical.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

from .llm import complete_messages, stream
from .schemas import ChatMessage, Chunk

_SYSTEM = (
    "You are a careful, grounded assistant. Answer the user's question using ONLY "
    "the provided context passages. Rules:\n"
    "- Cite every factual sentence with the passage number in square brackets, "
    "e.g. [1] or [2]. Only use the numbers shown in the context.\n"
    "- If the context does not contain the answer, say you don't have that "
    "information in the available documents. Do not invent facts.\n"
    "- Be concise and direct; no filler or small talk.\n"
    "- If passages conflict, note the discrepancy."
)


def _context_block(docs: List[Chunk]) -> str:
    # Number passages 1..N by position; this same numbering is used by the
    # answer verifier and the citation list so markers stay consistent.
    return "\n\n".join(
        f"[{i + 1}] (source: {d.citation_label()})\n{d.text}" for i, d in enumerate(docs)
    )


def _build_messages(
    query: str, docs: List[Chunk], history: Optional[List[ChatMessage]]
) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": _SYSTEM}]
    for m in (history or [])[-6:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append(
        {
            "role": "user",
            "content": f"Context passages:\n{_context_block(docs)}\n\nQuestion: {query}",
        }
    )
    return messages


def generate(
    query: str, docs: List[Chunk], history: Optional[List[ChatMessage]] = None
) -> str:
    return complete_messages(
        _build_messages(query, docs, history), temperature=0.2, max_tokens=900
    )


def generate_stream(
    query: str, docs: List[Chunk], history: Optional[List[ChatMessage]] = None
) -> Iterator[str]:
    yield from stream(_build_messages(query, docs, history), temperature=0.2, max_tokens=900)
