"""Contextual compression.

Extract only the sentences from each chunk that are relevant to the query,
trimming token usage before generation while preserving citations. Falls back
to the untouched chunk on any error (never lose evidence).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from pydantic import BaseModel

from .config import get_settings
from .llm import parse
from .schemas import Chunk


class _Extract(BaseModel):
    relevant: bool
    extracted: str


def compress_chunk(query: str, chunk: Chunk) -> Chunk:
    # Already short enough that compressing it wouldn't save meaningful tokens.
    if len(chunk.text) <= get_settings().compression_min_chars:
        return chunk
    try:
        result = parse(
            system=(
                "Extract ONLY the sentences from the passage that help answer the "
                "question. Preserve wording exactly. If nothing is relevant, set "
                "relevant=false. Do not summarize or add words."
            ),
            user=f"Question: {query}\n\nPassage:\n{chunk.text}",
            schema=_Extract,
            max_tokens=400,
        )
    except Exception:
        return chunk
    if not result.relevant or not result.extracted.strip():
        return chunk  # keep original rather than drop evidence
    out = chunk.model_copy()
    out.text = result.extracted.strip()
    return out


def compress(query: str, chunks: List[Chunk]) -> List[Chunk]:
    """Compress all chunks concurrently instead of one LLM call at a time."""
    if not chunks:
        return []
    with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
        return list(pool.map(lambda c: compress_chunk(query, c), chunks))
