"""Compute-budget interface — unavailable measures are marked, not fabricated."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ComputeBudget:
    max_model_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    max_steps: Optional[int] = None
    max_elapsed_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "max_elapsed_ms": self.max_elapsed_ms,
        }


@dataclass
class ComputeUsage:
    model_calls: int = 0
    tokens: Optional[int] = None
    steps: int = 0
    elapsed_ms: Optional[float] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "tokens": self.tokens if self.tokens is not None else "unavailable",
            "steps": self.steps,
            "elapsed_ms": self.elapsed_ms if self.elapsed_ms is not None else "unavailable",
            "notes": list(self.notes),
        }


class BudgetTracker:
    def __init__(self, budget: Optional[ComputeBudget] = None) -> None:
        self.budget = budget or ComputeBudget()
        self.usage = ComputeUsage()
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is not None:
            self.usage.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    def record_call(self) -> None:
        self.usage.model_calls += 1
        self.usage.steps += 1

    def within_budget(self) -> bool:
        b, u = self.budget, self.usage
        if b.max_model_calls is not None and u.model_calls > b.max_model_calls:
            return False
        if b.max_steps is not None and u.steps > b.max_steps:
            return False
        if b.max_elapsed_ms is not None and u.elapsed_ms is not None and u.elapsed_ms > b.max_elapsed_ms:
            return False
        return True
