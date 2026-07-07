"""Persistent hybrid index: FAISS (dense) + BM25 (sparse), sharing one chunk set.

On disk under `INDEX_DIR`:
  * index.faiss   — normalized embeddings, cosine similarity via inner product
  * chunks.jsonl  — one Chunk per line (text + provenance)
  * meta.json     — embedding model/dim, built-at, counts

BM25 is rebuilt in memory from chunks at load time (cheap, deterministic).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .config import get_settings
from .schemas import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HybridStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunks: List[Chunk] = []
        self._faiss: Optional[faiss.Index] = None
        self._bm25: Optional[BM25Okapi] = None

    # ---- build ------------------------------------------------------------ #
    @classmethod
    def build(cls, chunks: List[Chunk], embeddings: List[List[float]]) -> "HybridStore":
        store = cls()
        store.chunks = chunks
        mat = _normalize(np.asarray(embeddings, dtype="float32"))
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        store._faiss = index
        store._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        return store

    def save(self) -> None:
        out = Path(self.settings.index_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(out / "index.faiss"))
        with (out / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "embedding_model": self.settings.embedding_model,
                    "embedding_dim": self._faiss.d,
                    "count": len(self.chunks),
                },
                indent=2,
            )
        )

    # ---- load ------------------------------------------------------------- #
    @classmethod
    def load(cls) -> "HybridStore":
        store = cls()
        idx_dir = Path(store.settings.index_dir)
        faiss_path = idx_dir / "index.faiss"
        chunks_path = idx_dir / "chunks.jsonl"
        if not faiss_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found at {idx_dir}. Run `python -m rag.ingest` first."
            )
        store._faiss = faiss.read_index(str(faiss_path))
        store.chunks = [
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text().splitlines()
            if line.strip()
        ]
        store._bm25 = BM25Okapi([_tokenize(c.text) for c in store.chunks])
        return store

    @classmethod
    def exists(cls) -> bool:
        idx_dir = Path(get_settings().index_dir)
        return (idx_dir / "index.faiss").exists() and (idx_dir / "chunks.jsonl").exists()

    # ---- search ----------------------------------------------------------- #
    def vector_search(self, query_vec: List[float], k: int) -> List[Tuple[Chunk, float]]:
        q = _normalize(np.asarray([query_vec], dtype="float32"))
        scores, ids = self._faiss.search(q, min(k, len(self.chunks)))
        out: List[Tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out

    def bm25_search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    @property
    def size(self) -> int:
        return len(self.chunks)
