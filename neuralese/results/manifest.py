"""Versioned run manifest and result writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuralese._version import __software_id__, __version__

MANIFEST_SCHEMA_ID = "neuralese.run_manifest"
MANIFEST_SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(
    *,
    experiment_id: str,
    seed: int,
    adapter_identity: Dict[str, Any],
    tensorgate_status: Dict[str, Any],
    conditions: List[str],
    dataset_id: str,
    split_id: str,
    budget: Dict[str, Any],
    runs: List[Dict[str, Any]],
    aggregates: Dict[str, Any],
    hypotheses: Optional[List[str]] = None,
    limitations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "software_id": __software_id__,
        "neuralese_version": __version__,
        "timestamp_utc": utc_now(),
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "seed": seed,
        "adapter": adapter_identity,
        "tensorgate": tensorgate_status,
        "conditions": conditions,
        "budget": budget,
        "hypotheses": hypotheses or [],
        "limitations": limitations or [],
        "runs": runs,
        "aggregates": aggregates,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_summary(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Neuralese run: {manifest.get('experiment_id')}",
        "",
        f"- Schema: `{manifest.get('schema_id')}` v{manifest.get('schema_version')}",
        f"- Software: `{manifest.get('software_id')}`",
        f"- Seed: `{manifest.get('seed')}`",
        f"- Dataset/split: `{manifest.get('dataset_id')}` / `{manifest.get('split_id')}`",
        f"- Timestamp (UTC): {manifest.get('timestamp_utc')}",
        "",
        "## TensorGate",
        "",
        f"```json\n{json.dumps(manifest.get('tensorgate', {}), indent=2)}\n```",
        "",
        "## Aggregates by condition",
        "",
    ]
    aggs = manifest.get("aggregates") or {}
    for cond, stats in sorted(aggs.items()):
        acc = stats.get("accuracy")
        acc_s = f"{acc:.3f}" if isinstance(acc, float) else str(acc)
        lines.append(
            f"- **{cond}**: accuracy={acc_s} "
            f"(n={stats.get('n_total')}, correct={stats.get('n_correct')}, "
            f"wrong={stats.get('n_wrong')}, inconclusive={stats.get('n_inconclusive')})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This synthetic pipeline smoke test validates the experiment harness,",
            "interventions, and TensorGate boundary — **not** that real language models",
            "benefit from latent-state reasoning.",
            "",
            "## Limitations",
            "",
        ]
    )
    for lim in manifest.get("limitations") or []:
        lines.append(f"- {lim}")
    path.write_text("\n".join(lines) + "\n")
