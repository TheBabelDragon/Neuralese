"""Deterministic local toy adapter for CI and pipeline smoke tests.

Does NOT demonstrate real neural latent reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from neuralese.core.types import Prediction, ReasoningTask
from neuralese.tensorgate_adapter.bridge import (
    LatentBundle,
    make_latent_bundle,
    tensorgate_available,
)


_SYMBOL_TABLE = {
    "RED": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "BLUE": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "GREEN": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "SQUARE": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "CIRCLE": np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
    "TRIANGLE": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
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


@dataclass
class ToyAdapter:
    adapter_id: str = "neuralese-toy"
    adapter_version: str = "0.1.0"
    model_id: str = "toy-symbol-encoder"
    model_revision: str = "v1"

    def identity(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "decoding": "nearest-symbol-L2",
            "note": "Toy adapter for CI only — not real neural latent reasoning",
        }

    def direct_answer(self, task: ReasoningTask) -> Prediction:
        ans = " ".join(
            str(task.intermediate_slots[k]).upper() for k in sorted(task.intermediate_slots)
        )
        return Prediction(
            task_id=task.task_id, answer=ans, confidence=1.0, raw={"mode": "direct"}
        )

    def text_intermediate(self, task: ReasoningTask) -> tuple:
        text = " | ".join(f"{k}={v}" for k, v in sorted(task.intermediate_slots.items()))
        ans = " ".join(
            str(task.intermediate_slots[k]).upper() for k in sorted(task.intermediate_slots)
        )
        return text, Prediction(
            task_id=task.task_id,
            answer=ans,
            confidence=1.0,
            raw={"mode": "text", "intermediate": text},
        )

    def produce_latent(self, task: ReasoningTask):
        arr = _encode_slots(task.intermediate_slots)
        if not tensorgate_available():
            return arr
        return make_latent_bundle(
            arr,
            model_id=self.model_id,
            model_revision=self.model_revision,
            layer_id="toy.intermediate",
            representation_semantics="synthetic_symbol_slots",
            architecture_id="toy-symbol-encoder",
        )

    def predict_from_latent(self, task: ReasoningTask, latent: Any) -> Prediction:
        if isinstance(latent, LatentBundle):
            arr = latent.array
            lid = latent.latent_id
        else:
            arr = np.asarray(latent)
            lid = ""
        keys = sorted(task.intermediate_slots.keys())
        ans = _decode_answer_from_latent(arr, keys)
        return Prediction(
            task_id=task.task_id,
            answer=ans,
            confidence=1.0 if "?" not in ans else 0.0,
            raw={"mode": "latent", "latent_id": lid},
        )
