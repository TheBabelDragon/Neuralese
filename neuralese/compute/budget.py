"""Compute-budget interface — unavailable measures are marked, not fabricated."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class BudgetExhausted(Exception):
    def __init__(self, message: str, usage: Dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass
class ComputeBudget:
    max_adapter_invocations: Optional[int] = None
    max_model_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    max_steps: Optional[int] = None
    max_elapsed_ms: Optional[float] = None
    policy_id: str = "fixed_max_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "max_adapter_invocations": self.max_adapter_invocations,
            "max_model_calls": self.max_model_calls,
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "max_elapsed_ms": self.max_elapsed_ms,
            "note": (
                "Configured maxima are policy limits, not evidence that conditions "
                "consumed equal compute."
            ),
        }


@dataclass
class ComputeUsage:
    adapter_invocations: int = 0
    model_calls: int = 0
    tokens: Optional[int] = None
    steps: int = 0
    elapsed_ms: Optional[float] = None
    synthetic: bool = False
    notes: list = field(default_factory=list)
    exhausted: bool = False
    exhaustion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_invocations": self.adapter_invocations,
            "model_calls": self.model_calls,
            "tokens": self.tokens if self.tokens is not None else "unavailable",
            "steps": self.steps,
            "elapsed_ms": self.elapsed_ms if self.elapsed_ms is not None else "unavailable",
            "synthetic": self.synthetic,
            "exhausted": self.exhausted,
            "exhaustion_reason": self.exhaustion_reason,
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

    def record(
        self,
        *,
        adapter_invocations: int = 0,
        model_calls: int = 0,
        tokens: Optional[int] = None,
        steps: int = 0,
        synthetic: bool = False,
    ) -> None:
        self.usage.adapter_invocations += adapter_invocations
        self.usage.model_calls += model_calls
        self.usage.steps += steps
        if synthetic:
            self.usage.synthetic = True
        if tokens is not None:
            if self.usage.tokens is None:
                self.usage.tokens = 0
            self.usage.tokens += tokens
        self._check_limits()

    def _check_limits(self) -> None:
        b, u = self.budget, self.usage
        if b.max_adapter_invocations is not None and u.adapter_invocations > b.max_adapter_invocations:
            u.exhausted = True
            u.exhaustion_reason = "max_adapter_invocations"
            raise BudgetExhausted("adapter invocation budget exhausted", u.to_dict())
        if b.max_model_calls is not None and u.model_calls > b.max_model_calls:
            u.exhausted = True
            u.exhaustion_reason = "max_model_calls"
            raise BudgetExhausted("model-call budget exhausted", u.to_dict())
        if b.max_steps is not None and u.steps > b.max_steps:
            u.exhausted = True
            u.exhaustion_reason = "max_steps"
            raise BudgetExhausted("step budget exhausted", u.to_dict())
        if b.max_tokens is not None and u.tokens is not None and u.tokens > b.max_tokens:
            u.exhausted = True
            u.exhaustion_reason = "max_tokens"
            raise BudgetExhausted("token budget exhausted", u.to_dict())
        if self._t0 is not None and b.max_elapsed_ms is not None:
            elapsed = (time.perf_counter() - self._t0) * 1000.0
            u.elapsed_ms = elapsed
            if elapsed > b.max_elapsed_ms:
                u.exhausted = True
                u.exhaustion_reason = "max_elapsed_ms"
                raise BudgetExhausted("elapsed-time budget exhausted", u.to_dict())

    def within_budget(self) -> bool:
        try:
            self._check_limits()
            return not self.usage.exhausted
        except BudgetExhausted:
            return False
