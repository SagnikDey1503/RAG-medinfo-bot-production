"""Retrieval trace / audit log.

Every request produces a Trace: an ordered list of steps (query rewrite,
retrieval, verification, generation, ...) with timings and structured data.
Traces are returned to the client for transparency AND persisted to disk for
debugging and compliance (requirement #25 in the roadmap).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_settings
from .schemas import TraceStep


class Trace:
    def __init__(self, query: str) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.query = query
        self.created_at = time.time()
        self.steps: List[TraceStep] = []

    def add(
        self,
        name: str,
        detail: str = "",
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps.append(
            TraceStep(name=name, detail=detail, duration_ms=duration_ms, data=data)
        )

    def as_steps(self) -> List[TraceStep]:
        return self.steps

    def persist(self, extra: Optional[Dict[str, Any]] = None) -> None:
        settings = get_settings()
        out_dir = Path(settings.trace_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "trace_id": self.trace_id,
                "query": self.query,
                "created_at": self.created_at,
                "steps": [s.model_dump() for s in self.steps],
                "extra": extra or {},
            }
            (out_dir / f"{self.trace_id}.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
        except Exception:
            # Tracing must never break the request path.
            pass
