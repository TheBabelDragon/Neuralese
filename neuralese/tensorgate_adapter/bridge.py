"""TensorGate integration boundary.

Uses TensorGate's public latent-state API when available.
Does not duplicate TensorGate responsibilities or invent APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from neuralese._version import __software_id__

_TENSORGATE_AVAILABLE = False
_TENSORGATE_ERROR: Optional[str] = None

try:
    from tensorgate import (  # type: ignore
        LATENT_SCHEMA_ID,
        LATENT_SCHEMA_VERSION,
        LatentCompatibilityLevel,
        LatentStateSpec,
        check_latent_compatibility,
        describe_latent_state,
        deserialize_latent_state,
        latent_identity,
        serialize_latent_state,
    )

    _TENSORGATE_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - environment dependent
    _TENSORGATE_ERROR = str(exc)
    LATENT_SCHEMA_ID = "tensorgate.latent_state"
    LATENT_SCHEMA_VERSION = "1.0.0"
    LatentCompatibilityLevel = None  # type: ignore
    LatentStateSpec = None  # type: ignore


def tensorgate_available() -> bool:
    return _TENSORGATE_AVAILABLE


def tensorgate_status() -> Dict[str, Any]:
    return {
        "available": _TENSORGATE_AVAILABLE,
        "error": _TENSORGATE_ERROR,
        "schema_id": LATENT_SCHEMA_ID,
        "schema_version": LATENT_SCHEMA_VERSION,
        "consumer": __software_id__,
    }


@dataclass
class LatentBundle:
    """Array + TensorGate-compatible metadata envelope."""

    array: np.ndarray
    spec: Any  # LatentStateSpec when TensorGate present
    latent_id: str
    envelope: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "shape": list(self.array.shape),
            "dtype": str(self.array.dtype),
            "spec": self.spec.to_dict() if hasattr(self.spec, "to_dict") else {},
            "envelope_schema": self.envelope.get("schema_id"),
            "envelope_version": self.envelope.get("schema_version"),
        }


def require_tensorgate() -> None:
    if not _TENSORGATE_AVAILABLE:
        raise RuntimeError(
            "TensorGate is not installed or not importable. "
            "Install with: pip install tensorgate  "
            f"(import error: {_TENSORGATE_ERROR})"
        )


def make_latent_bundle(
    array: np.ndarray,
    *,
    model_id: str,
    layer_id: str,
    representation_semantics: str,
    model_revision: str = "",
    architecture_id: str = "neuralese-toy",
) -> LatentBundle:
    """Wrap an array in a validated TensorGate latent-state bundle."""
    require_tensorgate()
    arr = np.asarray(array)
    spec = LatentStateSpec(
        shape=tuple(int(x) for x in arr.shape),
        dtype=str(arr.dtype),
        layout="C",
        representation_semantics=representation_semantics,
        model_id=model_id,
        model_revision=model_revision,
        architecture_id=architecture_id,
        layer_id=layer_id,
    )
    desc = describe_latent_state(arr, spec)
    envelope = serialize_latent_state(arr, spec, label=layer_id)
    return LatentBundle(
        array=arr,
        spec=spec,
        latent_id=desc.latent_id,
        envelope=envelope,
    )


def validate_bundle(bundle: LatentBundle) -> Tuple[bool, str]:
    """Round-trip validate via TensorGate deserialize (fail-closed)."""
    require_tensorgate()
    try:
        arr, spec, ident = deserialize_latent_state(bundle.envelope, require_data=True)
        if arr is None:
            return False, "missing array after deserialize"
        if ident.content_hash != bundle.latent_id:
            return False, "latent identity mismatch after round-trip"
        if tuple(arr.shape) != tuple(bundle.array.shape):
            return False, "shape mismatch after round-trip"
        return True, "ok"
    except Exception as exc:
        return False, f"validation failed: {exc}"


def check_bundles_compatible(
    a: LatentBundle, b: LatentBundle, *, allow_dtype_conversion: bool = False
) -> Dict[str, Any]:
    """Compatibility via TensorGate; matching shape alone is never enough."""
    require_tensorgate()
    report = check_latent_compatibility(
        a.spec, b.spec, allow_dtype_conversion=allow_dtype_conversion
    )
    return report.to_dict()
