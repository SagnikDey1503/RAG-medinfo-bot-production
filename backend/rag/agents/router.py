"""Router / planner agent.

Decides whether a query needs document retrieval at all, whether it is
multi-hop (needs decomposition), and its rough domain. Cheap gate that saves
work on chit-chat and routes complex questions to the decomposition path.
"""
from __future__ import annotations

from ..config import get_settings
from ..llm import parse
from ..schemas import RouteDecision

_SYSTEM = (
    "You are a routing agent for a retrieval-augmented assistant grounded in a "
    "document corpus (e.g. a medical encyclopedia). Classify the user's message:\n"
    "- needs_retrieval: false ONLY for greetings, thanks, or meta questions about "
    "the assistant itself; true for anything answerable from documents.\n"
    "- is_multi_hop: true if answering requires combining multiple distinct facts "
    "or sub-questions.\n"
    "- domain: a short topical label."
)


def route(query: str) -> RouteDecision:
    settings = get_settings()
    if not settings.enable_router:
        return RouteDecision(needs_retrieval=True, is_multi_hop=False, domain="general")
    try:
        return parse(_SYSTEM, f"User message: {query}", RouteDecision, max_tokens=300)
    except Exception:
        # Fail open: assume we need retrieval.
        return RouteDecision(
            needs_retrieval=True, is_multi_hop=False, domain="general",
            reasoning="router-fallback",
        )
