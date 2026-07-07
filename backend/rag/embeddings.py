"""OpenAI embedding helpers with batching."""
from __future__ import annotations

from typing import List

from .config import get_settings
from .llm import get_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts, batching to respect API limits."""
    settings = get_settings()
    client = get_client()
    out: List[List[float]] = []
    batch = settings.embed_batch_size
    for i in range(0, len(texts), batch):
        chunk = [t.replace("\n", " ") for t in texts[i : i + batch]]
        resp = client.embeddings.create(model=settings.embedding_model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
