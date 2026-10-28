"""Tests for synthetic pipeline smoke experiment."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neuralese.adapters.toy import ToyAdapter
from neuralese.experiments.synthetic_pipeline import (
    SYNTHETIC_EXPERIMENT_ID,
    make_demo_tasks,
    run_synthetic_pipeline_smoke,
)
from neuralese.interventions.ops import ablate_latent, shuffle_latent
from neuralese.metrics.eval import exact_match
from neuralese.core.types import ReferenceAnswer
from neuralese.tensorgate_adapter.bridge import (
    check_bundles_compatible,
    make_latent_bundle,
    tensorgate_available,
    validate_bundle,
)


def test_intact_latent_supports_task():
    adapter = ToyAdapter()
    tasks = make_demo_tasks()
    for t in tasks:
        latent = adapter.produce_latent(t)
        pred = adapter.predict_from_latent(t, latent)
        ref = ReferenceAnswer(
            task_id=t.task_id,
            answer=" ".join(
                str(t.intermediate_slots[k]).upper() for k in sorted(t.intermediate_slots)
            ),
        )
        out = exact_match(pred, ref)
        assert out.correct is True


def test_shuffle_changes_performance():
    adapter = ToyAdapter()
    t = make_demo_tasks()[0]
    latent = adapter.produce_latent(t)
    mod, rec = shuffle_latent(latent, seed=0)
    pred = adapter.predict_from_latent(t, mod)
    ref = ReferenceAnswer(
        task_id=t.task_id,
        answer=" ".join(
            str(t.intermediate_slots[k]).upper() for k in sorted(t.intermediate_slots)
        ),
    )
    out = exact_match(pred, ref)
    assert out.correct is False
    assert rec["op"] == "shuffle"


def test_ablation_changes_performance():
    adapter = ToyAdapter()
    t = make_demo_tasks()[0]
    latent = adapter.produce_latent(t)
    mod, rec = ablate_latent(latent, mode="zero")
    pred = adapter.predict_from_latent(t, mod)
    ref = ReferenceAnswer(
        task_id=t.task_id,
        answer=" ".join(
            str(t.intermediate_slots[k]).upper() for k in sorted(t.intermediate_slots)
        ),
    )
    out = exact_match(pred, ref)
    assert out.correct is False
    assert rec["op"] == "ablate"


def test_irrelevant_soft_corrupt_not_automatic_causal_success():
    from neuralese.interventions.ops import corrupt_latent

    adapter = ToyAdapter()
    t = make_demo_tasks()[0]
    latent = adapter.produce_latent(t)
    mod, rec = corrupt_latent(latent, seed=1, scale=1e-6)
    pred = adapter.predict_from_latent(t, mod)
    ref = ReferenceAnswer(
        task_id=t.task_id,
        answer=" ".join(
            str(t.intermediate_slots[k]).upper() for k in sorted(t.intermediate_slots)
        ),
    )
    out = exact_match(pred, ref)
    assert out.correct is True
    assert rec["op"] == "corrupt"


def test_reproducible_seed():
    m1 = run_synthetic_pipeline_smoke(seed=123)
    m2 = run_synthetic_pipeline_smoke(seed=123)
    assert m1["aggregates"] == m2["aggregates"]
    assert m1["seed"] == m2["seed"] == 123


def test_manifest_schema():
    m = run_synthetic_pipeline_smoke(seed=42)
    assert m["schema_id"] == "neuralese.run_manifest"
    assert m["experiment_id"] == SYNTHETIC_EXPERIMENT_ID
    assert "direct_answer" in m["aggregates"]
    assert m["aggregates"]["direct_answer"]["n_total"] == 3
    assert m["aggregates"]["direct_answer"]["accuracy"] == 1.0
    assert m["aggregates"]["latent_intermediate"]["accuracy"] == 1.0
    assert m["aggregates"]["latent_shuffled"]["accuracy"] < 1.0
    assert m["aggregates"]["latent_ablated"]["accuracy"] < 1.0


@pytest.mark.skipif(not tensorgate_available(), reason="TensorGate not installed")
def test_tensorgate_bundle_validation():
    arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = make_latent_bundle(
        arr,
        model_id="toy",
        layer_id="l0",
        representation_semantics="test",
    )
    ok, detail = validate_bundle(b)
    assert ok, detail


@pytest.mark.skipif(not tensorgate_available(), reason="TensorGate not installed")
def test_incompatible_metadata_not_silent():
    a = make_latent_bundle(
        np.ones(4, dtype=np.float32),
        model_id="A",
        layer_id="l0",
        representation_semantics="s",
    )
    b = make_latent_bundle(
        np.ones(4, dtype=np.float32),
        model_id="B",
        layer_id="l0",
        representation_semantics="s",
    )
    report = check_bundles_compatible(a, b)
    assert report["compatible"] is False
    assert report["level"] == "incompatible"


def test_artifacts_written(tmp_path):
    m = run_synthetic_pipeline_smoke(seed=7, output_dir=str(tmp_path))
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.md").exists()
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["seed"] == 7
    assert data["experiment_id"] == m["experiment_id"]
