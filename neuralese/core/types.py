"""Typed protocols and data structures for reasoning tasks and runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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
class ReasoningRun:
    run_id: str
    experiment_id: str
    condition: ExperimentCondition
    status: RunStatus
    seed: int
    predictions: List[Prediction] = field(default_factory=list)
    outcomes: List[EvaluationOutcome] = field(default_factory=list)
    latent_ids: List[str] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def identity(self) -> Dict[str, Any]: ...
    def direct_answer(self, task: ReasoningTask) -> Prediction: ...
    def text_intermediate(self, task: ReasoningTask) -> tuple: ...
    def produce_latent(self, task: ReasoningTask) -> Any: ...
    def predict_from_latent(self, task: ReasoningTask, latent: Any) -> Prediction: ...
