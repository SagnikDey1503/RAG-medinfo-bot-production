"""Thin wrapper around the OpenAI SDK.

Centralizes client creation, structured-output parsing, plain completions and
streaming so the rest of the codebase never touches the SDK directly. This
makes it trivial to swap providers or add retry/caching later.
"""
from __future__ import annotations

import time
from typing import Iterator, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to backend/.env before starting."
            )
        _client = OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)
    return _client


def _messages(system: str, user: str) -> List[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def complete(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """Plain text completion from a single system + user turn."""
    return complete_messages(
        _messages(system, user), model=model, temperature=temperature, max_tokens=max_tokens
    )


def complete_messages(
    messages: List[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """Plain text completion from a full message list (preserves history)."""
    settings = get_settings()
    resp = get_client().chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def parse(
    system: str,
    user: str,
    schema: Type[T],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> T:
    """Structured output: forces the model to return an instance of `schema`.

    Uses OpenAI structured outputs (JSON schema) via the `parse` helper. On any
    failure the caller is expected to handle the exception and degrade
    gracefully — never let a single agent hiccup break the whole pipeline.
    """
    settings = get_settings()
    completion = get_client().beta.chat.completions.parse(
        model=model or settings.llm_model,
        messages=_messages(system, user),
        response_format=schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Structured output returned no parsed object")
    return parsed


def stream(
    messages: List[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Iterator[str]:
    """Yield answer tokens as they arrive."""
    settings = get_settings()
    response = get_client().chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        # Some chunks (keepalives / usage frames) carry no choices or delta.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


class Timer:
    """Tiny context manager returning elapsed milliseconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
