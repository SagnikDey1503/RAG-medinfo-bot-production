"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Given one or more query strings (the original plus rewriter variants + a HyDE
document), run dense (FAISS) and sparse (BM25) search for each, then fuse all
result lists into a single ranking with RRF. RRF is score-scale agnostic, which
is exactly what we need when mixing cosine similarity with BM25 scores.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import get_settings
from .embeddings import embed_texts
from .schemas import Chunk
from .vectorstore import HybridStore


class HybridRetriever:
    def __init__(self, store: HybridStore) -> None:
        self.store = store
        self.settings = get_settings()

    def _ranked_lists(
        self, queries: List[str], hyde_docs: Optional[List[str]] = None
    ) -> List[List[Tuple[str, float]]]:
        """Produce ranked (chunk_id, score) lists from every retriever/query."""
        lists: List[List[Tuple[str, float]]] = []
        s = self.settings

        # Dense search uses embeddings of the query text; HyDE docs are embedded
        # as if they were the query (hypothetical answer -> better dense recall).
        # Embed everything in ONE batched API call instead of one call per variant.
        dense_inputs = [q for q in (list(queries) + list(hyde_docs or [])) if q.strip()]
        if dense_inputs:
            vectors = embed_texts(dense_inputs)
            for vec in vectors:
                lists.append(
                    [(c.id, sc) for c, sc in self.store.vector_search(vec, s.vector_k)]
                )

        # Sparse (BM25) search only over literal query strings.
        for q in queries:
            if not q.strip():
                continue
            lists.append(
                [(c.id, sc) for c, sc in self.store.bm25_search(q, s.bm25_k)]
            )
        return lists

    def retrieve(
        self, queries: List[str], hyde_docs: Optional[List[str]] = None
    ) -> List[Chunk]:
        s = self.settings
        ranked_lists = self._ranked_lists(queries, hyde_docs)

        by_id: Dict[str, Chunk] = {c.id: c for c in self.store.chunks}
        fused: Dict[str, float] = {}
        for ranking in ranked_lists:
            for rank, (cid, _score) in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (s.rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: s.fusion_k]
        out: List[Chunk] = []
        for cid, score in ordered:
            chunk = by_id[cid].model_copy()
            chunk.fused_score = round(score, 6)
            out.append(chunk)
        return out
