"""CRAG web-search fallback.

When corpus retrieval is judged insufficient, fetch fresh evidence from the web
(Tavily). Results are wrapped as `Chunk`s with origin="web" so they flow through
reranking, generation and citation exactly like corpus chunks. Disabled unless
TAVILY_API_KEY is set.
"""
from __future__ import annotations

import hashlib
from typing import List

from ..config import get_settings
from ..schemas import Chunk


def _web_id(url: str, idx: int) -> str:
    return "web-" + hashlib.sha1(f"{url}:{idx}".encode()).hexdigest()[:10]


def search(query: str, max_results: int = 5) -> List[Chunk]:
    settings = get_settings()
    if not settings.web_search_enabled:
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(
            query=query, max_results=max_results, search_depth="advanced"
        )
    except Exception:
        return []

    out: List[Chunk] = []
    for i, r in enumerate(resp.get("results", [])):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        url = r.get("url", "")
        out.append(
            Chunk(
                id=_web_id(url, i),
                text=content,
                source=url or "web",
                page=None,
                origin="web",
            )
        )
    return out
