"""Typed protocols and data structures for reasoning tasks and runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SKIPPED = "skipped"


class ConditionOutcome(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


class HypothesisResult(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


class RunStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    SKIPPED = "skipped"


class ExperimentCondition(str, Enum):
    DIRECT = "direct_answer"
    TEXT_INTERMEDIATE = "text_intermediate"
    LATENT_INTERMEDIATE = "latent_intermediate"
    LATENT_SHUFFLED = "latent_shuffled"
    LATENT_ABLATED = "latent_ablated"
    LATENT_NOISE_CONTROL = "latent_noise_control"


@dataclass
class ReasoningTask:
    task_id: str
    input_text: str
    intermediate_slots: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceAnswer:
    task_id: str
    answer: str
    source: str = "dataset"


@dataclass
class Prediction:
    task_id: str
    answer: str
    confidence: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationOutcome:
    task_id: str
    correct: Optional[bool]
    status: RunStatus
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationCompleteness:
    n_expected: int = 0
    n_completed: int = 0
    n_failed: int = 0
    n_inconclusive: int = 0
    n_error: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_expected": self.n_expected,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "n_inconclusive": self.n_inconclusive,
            "n_error": self.n_error,
        }


@dataclass
class ReasoningRun:
    run_id: str
    experiment_id: str
    condition: ExperimentCondition
    seed: int
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED
    condition_outcome: ConditionOutcome = ConditionOutcome.NOT_EVALUATED
    hypothesis_result: HypothesisResult = HypothesisResult.NOT_EVALUATED
    completeness: EvaluationCompleteness = field(default_factory=EvaluationCompleteness)
    predictions: List[Prediction] = field(default_factory=list)
    outcomes: List[EvaluationOutcome] = field(default_factory=list)
    latent_ids: List[str] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)
    status: RunStatus = RunStatus.PASSED


@dataclass
class AdapterCapabilities:
    direct_answer: bool = False
    text_intermediate: bool = False
    latent_produce: bool = False
    latent_consume: bool = False
    reports_model_calls: bool = False
    reports_tokens: bool = False
    reports_latency: bool = False
    synthetic_accounting: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direct_answer": self.direct_answer,
            "text_intermediate": self.text_intermediate,
            "latent_produce": self.latent_produce,
            "latent_consume": self.latent_consume,
            "reports_model_calls": self.reports_model_calls,
            "reports_tokens": self.reports_tokens,
            "reports_latency": self.reports_latency,
            "synthetic_accounting": self.synthetic_accounting,
            "notes": self.notes,
        }


@dataclass
class UsageReport:
    adapter_invocations: int = 1
    model_calls: int = 0
    tokens: Optional[int] = None
    steps: int = 0
    latency_ms: Optional[float] = None
    synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_invocations": self.adapter_invocations,
            "model_calls": self.model_calls,
            "tokens": self.tokens if self.tokens is not None else "unavailable",
            "steps": self.steps,
            "latency_ms": self.latency_ms if self.latency_ms is not None else "unavailable",
            "synthetic": self.synthetic,
        }


@runtime_checkable
class ModelAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def identity(self) -> Dict[str, Any]: ...
    def capabilities(self) -> AdapterCapabilities: ...
    def direct_answer(self, task: ReasoningTask) -> tuple: ...
    def text_intermediate(self, task: ReasoningTask) -> tuple: ...
    def produce_latent(self, task: ReasoningTask) -> tuple: ...
    def predict_from_latent(self, task: ReasoningTask, latent: Any) -> tuple: ...
