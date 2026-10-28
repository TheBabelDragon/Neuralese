"""Stable real-model adapter boundary.

Core remains free of GPU / paid-API dependencies.
Adapters that cannot expose latents must report unsupported capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from neuralese.core.types import AdapterCapabilities, Prediction, ReasoningTask, UsageReport


class UnsupportedOperation(Exception):
    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(f"unsupported operation {operation!r}: {reason}")
        self.operation = operation
        self.reason = reason


@runtime_checkable
class RealModelAdapter(Protocol):
    """Optional real-model integration surface.

    Never describe simulated vectors as real hidden states.
    Raise UnsupportedOperation when latent produce/consume is unavailable.
    """

    adapter_id: str
    adapter_version: str

    def identity(self) -> Dict[str, Any]: ...
    def capabilities(self) -> AdapterCapabilities: ...
    def representation_metadata(self) -> Dict[str, Any]: ...
    def direct_answer(self, task: ReasoningTask) -> tuple: ...
    def text_intermediate(self, task: ReasoningTask) -> tuple: ...
    def produce_latent(self, task: ReasoningTask) -> tuple: ...
    def predict_from_latent(self, task: ReasoningTask, latent: Any) -> tuple: ...
