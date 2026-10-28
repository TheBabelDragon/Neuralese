"""Deterministic local toy adapter for CI and pipeline smoke tests.

Does NOT demonstrate real neural latent reasoning.
Accounting is synthetic and must never be reported as real model cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from neuralese.core.types import AdapterCapabilities, Prediction, ReasoningTask, UsageReport
from neuralese.tensorgate_adapter.bridge import (
    LatentBundle,
    make_latent_bundle,
    tensorgate_available,
)


_SYMBOL_TABLE = {
    "RED": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "BLUE": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "GREEN": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "YELLOW": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "SQUARE": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "CIRCLE": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    "TRIANGLE": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
    "HEXAGON": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
}


def _encode_slots(slots: Dict[str, Any]) -> np.ndarray:
    parts = []
    for key in sorted(slots.keys()):
        val = str(slots[key]).upper()
        parts.append(_SYMBOL_TABLE.get(val, np.zeros(4, dtype=np.float32)))
    if not parts:
        return np.zeros(4, dtype=np.float32)
    return np.concatenate(parts)


def _decode_answer_from_latent(arr: np.ndarray, slots_keys: list) -> str:
    arr = np.asarray(arr, dtype=np.float32).ravel()
    n_slots = max(1, len(slots_keys))
    width = 4
    recovered = []
    for i in range(n_slots):
        chunk = arr[i * width : (i + 1) * width]
        if chunk.size < width:
            recovered.append("?")
            continue
        best, best_d = "?", float("inf")
        for sym, vec in _SYMBOL_TABLE.items():
            d = float(np.linalg.norm(chunk - vec))
            if d < best_d:
                best_d = d
                best = sym
        recovered.append(best if best_d < 0.5 else "?")
    return " ".join(recovered)


def _synthetic_usage(*, steps: int = 1) -> UsageReport:
    return UsageReport(
        adapter_invocations=1,
        model_calls=0,
        tokens=None,
        steps=steps,
        latency_ms=None,
        synthetic=True,
    )


@dataclass
class ToyAdapter:
    adapter_id: str = "neuralese-toy"
    adapter_version: str = "0.2.0"
    model_id: str = "toy-symbol-encoder"
    model_revision: str = "v2"

    def identity(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "decoding": "nearest-symbol-L2",
            "note": "Toy adapter for CI only — not real neural latent reasoning",
            "accounting": "synthetic",
        }

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            direct_answer=True,
            text_intermediate=True,
            latent_produce=True,
            latent_consume=True,
            reports_model_calls=False,
            reports_tokens=False,
            reports_latency=False,
            synthetic_accounting=True,
            notes="Synthetic symbol encoder; model_calls always 0",
        )

    def direct_answer(self, task: ReasoningTask):
        ans = " ".join(
            str(task.intermediate_slots[k]).upper() for k in sorted(task.intermediate_slots)
        )
        pred = Prediction(
            task_id=task.task_id, answer=ans, confidence=1.0, raw={"mode": "direct"}
        )
        return pred, _synthetic_usage()

    def text_intermediate(self, task: ReasoningTask):
        text = " | ".join(f"{k}={v}" for k, v in sorted(task.intermediate_slots.items()))
        ans = " ".join(
            str(task.intermediate_slots[k]).upper() for k in sorted(task.intermediate_slots)
        )
        pred = Prediction(
            task_id=task.task_id,
            answer=ans,
            confidence=1.0,
            raw={"mode": "text", "intermediate": text},
        )
        return text, pred, _synthetic_usage()

    def produce_latent(self, task: ReasoningTask):
        arr = _encode_slots(task.intermediate_slots)
        usage = _synthetic_usage()
        if not tensorgate_available():
            return arr, usage
        bundle = make_latent_bundle(
            arr,
            model_id=self.model_id,
            model_revision=self.model_revision,
            layer_id="toy.intermediate",
            representation_semantics="synthetic_symbol_slots",
            architecture_id="toy-symbol-encoder",
        )
        return bundle, usage

    def predict_from_latent(self, task: ReasoningTask, latent: Any):
        if isinstance(latent, LatentBundle):
            arr = latent.array
            lid = latent.latent_id
        else:
            arr = np.asarray(latent)
            lid = ""
        keys = sorted(task.intermediate_slots.keys())
        ans = _decode_answer_from_latent(arr, keys)
        pred = Prediction(
            task_id=task.task_id,
            answer=ans,
            confidence=1.0 if "?" not in ans else 0.0,
            raw={"mode": "latent", "latent_id": lid},
        )
        return pred, _synthetic_usage()
