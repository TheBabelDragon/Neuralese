"""Controlled latent-state interventions with audit trail."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from neuralese.tensorgate_adapter.bridge import LatentBundle, make_latent_bundle, tensorgate_available


def _as_array(latent: Any) -> np.ndarray:
    if isinstance(latent, LatentBundle):
        return np.array(latent.array, copy=True)
    return np.array(latent, copy=True)


def _rewrap(arr: np.ndarray, original: Any, *, tag: str) -> Any:
    if isinstance(original, LatentBundle) and tensorgate_available():
        spec = original.spec
        return make_latent_bundle(
            arr,
            model_id=getattr(spec, "model_id", "") + f"/intervened:{tag}",
            model_revision=getattr(spec, "model_revision", ""),
            layer_id=getattr(spec, "layer_id", "") + f".{tag}",
            representation_semantics=getattr(spec, "representation_semantics", ""),
            architecture_id=getattr(spec, "architecture_id", ""),
        )
    return arr


def shuffle_latent(latent: Any, *, seed: int) -> tuple:
    arr = _as_array(latent)
    rng = np.random.default_rng(seed)
    flat = arr.ravel().copy()
    rng.shuffle(flat)
    out = flat.reshape(arr.shape)
    record = {
        "op": "shuffle",
        "seed": seed,
        "shape": list(arr.shape),
        "original_id": getattr(latent, "latent_id", None),
    }
    new = _rewrap(out, latent, tag="shuffle")
    if isinstance(new, LatentBundle):
        record["result_id"] = new.latent_id
    return new, record


def ablate_latent(latent: Any, *, mode: str = "zero") -> tuple:
    arr = _as_array(latent)
    if mode == "zero":
        out = np.zeros_like(arr)
    elif mode == "mean":
        out = np.full_like(arr, float(np.mean(arr)))
    else:
        raise ValueError(f"unknown ablation mode: {mode}")
    record = {
        "op": "ablate",
        "mode": mode,
        "shape": list(arr.shape),
        "original_id": getattr(latent, "latent_id", None),
    }
    new = _rewrap(out, latent, tag=f"ablate_{mode}")
    if isinstance(new, LatentBundle):
        record["result_id"] = new.latent_id
    return new, record


def replace_latent(latent: Any, replacement: np.ndarray) -> tuple:
    arr = _as_array(latent)
    rep = np.asarray(replacement, dtype=arr.dtype)
    if rep.shape != arr.shape:
        raise ValueError(f"replacement shape {rep.shape} != latent shape {arr.shape}")
    record = {
        "op": "replace",
        "shape": list(arr.shape),
        "original_id": getattr(latent, "latent_id", None),
    }
    new = _rewrap(rep, latent, tag="replace")
    if isinstance(new, LatentBundle):
        record["result_id"] = new.latent_id
    return new, record


def corrupt_latent(latent: Any, *, seed: int, scale: float = 1.0) -> tuple:
    arr = _as_array(latent)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, size=arr.shape).astype(arr.dtype)
    out = arr + noise
    record = {
        "op": "corrupt",
        "seed": seed,
        "scale": scale,
        "shape": list(arr.shape),
        "original_id": getattr(latent, "latent_id", None),
    }
    new = _rewrap(out, latent, tag="corrupt")
    if isinstance(new, LatentBundle):
        record["result_id"] = new.latent_id
    return new, record


def assert_ids_differ(original: Any, modified: Any) -> None:
    oid = getattr(original, "latent_id", None)
    mid = getattr(modified, "latent_id", None)
    if oid and mid and oid == mid:
        raise RuntimeError("intervention did not change latent identity")
