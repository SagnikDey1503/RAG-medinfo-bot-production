"""The agentic RAG pipeline.

Orchestrates every component into a self-healing loop:

    route -> (rewrite + multi-query + HyDE) -> hybrid retrieve -> rerank
          -> retrieval verifier --insufficient--> escalate / web fallback --+
                 |                                                           |
             sufficient <---------------------------------------------------+
                 |
          compress -> generate -> answer verifier (citation/hallucination)
          -> confidence -> trace

Designed to degrade gracefully: any single agent failure falls back to a sane
default rather than breaking the request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .agents import (
    answer_verifier,
    confidence as confidence_agent,
    query_rewriter,
    retrieval_verifier,
    router,
    web_search,
)
from .compress import compress
from .config import get_settings
from .generate import generate, generate_stream
from .rerank import rerank
from .retrieval import HybridRetriever
from .schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    RetrievalVerdict,
)
from .trace import Trace
from .llm import Timer
from .vectorstore import HybridStore

# Numeric citation markers the generator emits, e.g. [1], [2].
_CITE_NUM_RE = re.compile(r"\[(\d{1,3})\]")

_LOW_CONFIDENCE_NOTE = (
    "_I couldn't find strongly supporting information for this question in the "
    "documents, so treat the following with extra caution:_\n\n"
)


@dataclass
class RetrievalContext:
    docs: List[Chunk]
    verdict: RetrievalVerdict
    heal_attempts: int
    used_web: bool
    trace: Trace
    needs_retrieval: bool = True
    history: List[ChatMessage] = field(default_factory=list)
    query: str = ""


class RAGPipeline:
    def __init__(self, store: Optional[HybridStore] = None) -> None:
        self.settings = get_settings()
        self.store = store or HybridStore.load()
        self.retriever = HybridRetriever(self.store)

    # ------------------------------------------------------------------ #
    # Retrieval + self-healing                                           #
    # ------------------------------------------------------------------ #
    def build_context(self, request: ChatRequest) -> RetrievalContext:
        query = request.message.strip()
        history = request.history
        trace = Trace(query)

        # 1. Route
        with Timer() as t:
            decision = router.route(query)
        trace.add(
            "router", decision.reasoning, t.ms,
            {"needs_retrieval": decision.needs_retrieval, "multi_hop": decision.is_multi_hop},
        )
        if not decision.needs_retrieval:
            return RetrievalContext(
                docs=[], verdict=RetrievalVerdict(sufficient=True, score=1.0),
                heal_attempts=0, used_web=False, trace=trace,
                needs_retrieval=False, history=history, query=query,
            )

        best_docs: List[Chunk] = []
        best_verdict = RetrievalVerdict(sufficient=False, score=0.0)
        used_web = False
        attempts = 0

        for attempt in range(1, self.settings.max_heal_attempts + 1):
            attempts = attempt
            aggressive = attempt > 1

            # 2. Query expansion (rewrite + multi-query + HyDE + step-back + decompose)
            with Timer() as t:
                expansion = query_rewriter.expand(
                    query, history=history,
                    multi_hop=decision.is_multi_hop, aggressive=aggressive,
                )
            variants = query_rewriter.all_query_variants(query, expansion)
            hyde = [expansion.hyde_document] if expansion.hyde_document else []
            trace.add(
                f"query_expansion.attempt{attempt}",
                f"{len(variants)} variants, hyde={bool(hyde)}", t.ms,
                {"rewritten": expansion.rewritten, "variants": variants,
                 "step_back": expansion.step_back, "sub_queries": expansion.sub_queries},
            )

            # 3. Hybrid retrieve (RRF fusion across all variants + HyDE)
            with Timer() as t:
                candidates = self.retriever.retrieve(variants, hyde_docs=hyde)
            trace.add(f"hybrid_retrieval.attempt{attempt}", f"{len(candidates)} fused candidates", t.ms)

            # 4. Rerank (cross-encoder)
            if self.settings.enable_rerank:
                with Timer() as t:
                    docs = rerank(expansion.rewritten or query, candidates, self.settings.final_k)
                trace.add(f"rerank.attempt{attempt}", f"top {len(docs)}", t.ms,
                          {"scores": [round(d.rerank_score or 0, 3) for d in docs]})
            else:
                docs = candidates[: self.settings.final_k]

            # 5. Verify retrieval
            with Timer() as t:
                verdict = retrieval_verifier.verify(query, docs)
            trace.add(f"retrieval_verifier.attempt{attempt}",
                      f"score={verdict.score:.2f} sufficient={verdict.sufficient}", t.ms,
                      {"missing": verdict.missing_information})

            if verdict.score >= best_verdict.score:
                best_docs, best_verdict = docs, verdict
            if verdict.sufficient:
                break

        # 6. CRAG web fallback if still insufficient
        if not best_verdict.sufficient and self.settings.web_search_enabled:
            with Timer() as t:
                web_docs = web_search.search(query)
            if web_docs:
                used_web = True
                merged = best_docs + web_docs
                best_docs = rerank(query, merged, self.settings.final_k) \
                    if self.settings.enable_rerank else merged[: self.settings.final_k]
                best_verdict = retrieval_verifier.verify(query, best_docs)
                trace.add("web_fallback", f"{len(web_docs)} web results merged", t.ms,
                          {"new_score": best_verdict.score})

        return RetrievalContext(
            docs=best_docs, verdict=best_verdict, heal_attempts=attempts,
            used_web=used_web, trace=trace, needs_retrieval=True,
            history=history, query=query,
        )

    # ------------------------------------------------------------------ #
    # Full blocking answer                                               #
    # ------------------------------------------------------------------ #
    def answer(self, request: ChatRequest) -> ChatResponse:
        ctx = self.build_context(request)

        # Chit-chat / no-retrieval path.
        if not ctx.needs_retrieval:
            from .llm import complete_messages
            reply = complete_messages(
                self._chitchat_messages(ctx.query, ctx.history),
                temperature=0.4, max_tokens=300,
            )
            ctx.trace.add("direct_answer", "no retrieval needed")
            ctx.trace.persist()
            return ChatResponse(
                answer=reply, citations=[], confidence=0.9, confidence_label="high",
                used_web_search=False, heal_attempts=0,
                trace_id=ctx.trace.trace_id, trace=ctx.trace.as_steps(),
            )

        # Compression
        docs = ctx.docs
        if self.settings.enable_compression and docs:
            with Timer() as t:
                docs = compress(ctx.query, docs)
            ctx.trace.add("compression", f"compressed {len(docs)} passages", t.ms)

        # Generation
        with Timer() as t:
            draft = generate(ctx.query, docs, ctx.history)
        ctx.trace.add("generation", "draft produced", t.ms)

        # Answer verification (citation + hallucination)
        with Timer() as t:
            verification = answer_verifier.verify(ctx.query, docs, draft)
        final_answer = verification.corrected_answer
        ctx.trace.add(
            "answer_verifier",
            f"grounded={verification.grounded} removed={verification.unsupported_removed}",
            t.ms,
            {"claims": [c.model_dump() for c in verification.claims]},
        )

        # Confidence
        score, label = confidence_agent.estimate(
            ctx.verdict, verification, used_web=ctx.used_web, heal_attempts=ctx.heal_attempts,
        )
        ctx.trace.add("confidence", f"{label} ({score})")

        citations = self._citations(docs, final_answer)
        if not ctx.verdict.sufficient:
            final_answer = _LOW_CONFIDENCE_NOTE + final_answer
        ctx.trace.persist({"answer": final_answer, "confidence": score})

        return ChatResponse(
            answer=final_answer, citations=citations, confidence=score,
            confidence_label=label, used_web_search=ctx.used_web,
            heal_attempts=ctx.heal_attempts, trace_id=ctx.trace.trace_id,
            trace=ctx.trace.as_steps(),
        )

    # ------------------------------------------------------------------ #
    # Streaming answer (draft streamed, then verified correction event)  #
    # ------------------------------------------------------------------ #
    def stream_answer(self, request: ChatRequest):
        """Yield (event_type, payload) tuples for the API to serialize as SSE."""
        ctx = self.build_context(request)

        if not ctx.needs_retrieval:
            from .llm import stream as llm_stream
            acc = ""
            for d in llm_stream(
                self._chitchat_messages(ctx.query, ctx.history),
                temperature=0.4, max_tokens=300,
            ):
                acc += d
                yield ("token", d)
            ctx.trace.persist()
            yield ("done", ChatResponse(
                answer=acc, citations=[], confidence=0.9, confidence_label="high",
                used_web_search=False, heal_attempts=0,
                trace_id=ctx.trace.trace_id, trace=ctx.trace.as_steps(),
            ))
            return

        docs = ctx.docs
        if self.settings.enable_compression and docs:
            docs = compress(ctx.query, docs)

        # Surface a numbered source preview early so the UI can render it while streaming.
        yield ("meta", {"trace_id": ctx.trace.trace_id,
                        "citations": [c.model_dump() for c in self._citations(docs)]})

        if not ctx.verdict.sufficient:
            yield ("token", _LOW_CONFIDENCE_NOTE)

        draft = ""
        for token in generate_stream(ctx.query, docs, ctx.history):
            draft += token
            yield ("token", token)

        # Post-hoc verification: may correct the streamed draft. Fact-check only
        # the model's own draft — the caution note isn't a factual claim.
        verification = answer_verifier.verify(ctx.query, docs, draft)
        final_answer = verification.corrected_answer
        score, label = confidence_agent.estimate(
            ctx.verdict, verification, used_web=ctx.used_web, heal_attempts=ctx.heal_attempts,
        )
        ctx.trace.add("answer_verifier",
                      f"grounded={verification.grounded} removed={verification.unsupported_removed}")
        citations = self._citations(docs, final_answer)
        if not ctx.verdict.sufficient:
            final_answer = _LOW_CONFIDENCE_NOTE + final_answer
        ctx.trace.persist({"answer": final_answer, "confidence": score})

        yield ("done", ChatResponse(
            answer=final_answer, citations=citations,
            confidence=score, confidence_label=label, used_web_search=ctx.used_web,
            heal_attempts=ctx.heal_attempts, trace_id=ctx.trace.trace_id,
            trace=ctx.trace.as_steps(),
        ))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _chitchat_messages(query: str, history: List[ChatMessage]) -> List[dict]:
        msgs: List[dict] = [{
            "role": "system",
            "content": "You are a friendly, concise assistant for a document Q&A app. "
            "Answer briefly; if the user asks something factual, invite them to ask "
            "about the documents.",
        }]
        for m in (history or [])[-6:]:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": query})
        return msgs

    @staticmethod
    def _to_citation(n: int, d: Chunk) -> Citation:
        snippet = d.text.strip().replace("\n", " ")
        return Citation(
            id=str(n), label=d.citation_label(), source=d.source, page=d.page,
            snippet=snippet[:300] + ("..." if len(snippet) > 300 else ""),
            origin=d.origin,
        )

    @classmethod
    def _citations(cls, docs: List[Chunk], answer: Optional[str] = None) -> List[Citation]:
        """Build the numbered source list. Passages are numbered by position
        (matching the [n] markers the generator emits). When an answer is given,
        show only the passages it actually cited; otherwise show all retrieved
        passages (used for the early streaming preview, before the answer exists).
        """
        cited = {int(n) for n in _CITE_NUM_RE.findall(answer)} if answer else set()
        if cited:
            out = [cls._to_citation(i + 1, d) for i, d in enumerate(docs) if (i + 1) in cited]
            if out:
                return out
        return [cls._to_citation(i + 1, d) for i, d in enumerate(docs)]
