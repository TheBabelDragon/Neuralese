"""Tests for synthetic pipeline smoke experiment and hardening."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from neuralese.adapters.toy import ToyAdapter
from neuralese.compute.budget import BudgetExhausted, BudgetTracker, ComputeBudget
from neuralese.core.seeding import stable_seed, task_index_seed
from neuralese.core.types import ReferenceAnswer
from neuralese.experiments.synthetic_pipeline import (
    SYNTHETIC_EXPERIMENT_ID,
    make_demo_tasks,
    normalize_manifest_for_compare,
    run_synthetic_pipeline_smoke,
)
from neuralese.interventions.ops import ablate_latent, shuffle_latent
from neuralese.metrics.eval import exact_match
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
        latent, _ = adapter.produce_latent(t)
        pred, _ = adapter.predict_from_latent(t, latent)
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
    latent, _ = adapter.produce_latent(t)
    mod, rec = shuffle_latent(latent, seed=0)
    pred, _ = adapter.predict_from_latent(t, mod)
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
    latent, _ = adapter.produce_latent(t)
    mod, rec = ablate_latent(latent, mode="zero")
    pred, _ = adapter.predict_from_latent(t, mod)
    ref = ReferenceAnswer(
        task_id=t.task_id,
        answer=" ".join(
            str(t.intermediate_slots[k]).upper() for k in sorted(t.intermediate_slots)
        ),
    )
    out = exact_match(pred, ref)
    assert out.correct is False
    assert rec["op"] == "ablate"


def test_noise_control_preserves_performance():
    from neuralese.interventions.ops import corrupt_latent

    adapter = ToyAdapter()
    t = make_demo_tasks()[0]
    latent, _ = adapter.produce_latent(t)
    mod, rec = corrupt_latent(latent, seed=1, scale=1e-6)
    pred, _ = adapter.predict_from_latent(t, mod)
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
    assert normalize_manifest_for_compare(m1) == normalize_manifest_for_compare(m2)


def test_different_seeds_can_differ():
    m1 = normalize_manifest_for_compare(run_synthetic_pipeline_smoke(seed=1))
    m2 = normalize_manifest_for_compare(run_synthetic_pipeline_smoke(seed=2))
    assert m1["intervention_seeds"] != m2["intervention_seeds"]


def test_manifest_schema():
    m = run_synthetic_pipeline_smoke(seed=42)
    assert m["schema_id"] == "neuralese.run_manifest"
    assert m["experiment_id"] == SYNTHETIC_EXPERIMENT_ID
    assert m["n_tasks"] == 8
    assert m["aggregates"]["direct_answer"]["accuracy"] == 1.0
    assert m["aggregates"]["latent_intermediate"]["accuracy"] == 1.0
    assert m["aggregates"]["latent_shuffled"]["accuracy"] < 0.5
    assert m["aggregates"]["latent_ablated"]["accuracy"] < 0.5
    assert m["aggregates"]["latent_noise_control"]["accuracy"] == 1.0
    assert m["aggregates"]["latent_shuffled"]["execution_status"] == "completed"
    assert m["aggregates"]["latent_shuffled"]["hypothesis_result"] == "supported"
    assert m["aggregates"]["direct_answer"]["hypothesis_result"] == "supported"


def test_stable_seed_cross_process_algorithm():
    a = stable_seed(42, "shuffle", 0)
    b = stable_seed(42, "shuffle", 0)
    assert a == b
    assert task_index_seed(42, 0, "shuffle") == a
    assert stable_seed(42, "shuffle", 1) != a


def test_subprocess_reproducibility():
    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, ".")
        from neuralese.experiments.synthetic_pipeline import (
            run_synthetic_pipeline_smoke,
            normalize_manifest_for_compare,
        )
        m = run_synthetic_pipeline_smoke(seed=99)
        print(json.dumps(normalize_manifest_for_compare(m), sort_keys=True))
        """
    )
    env_base = {"PYTHONPATH": str(Path.cwd()) + ":/home/workdir/artifacts/TensorGate"}
    results = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            env={**dict(**__import__("os").environ), **env_base},
        )
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout.strip()))
    assert results[0] == results[1]


def test_budget_enforcement():
    tracker = BudgetTracker(ComputeBudget(max_adapter_invocations=2))
    tracker.record(adapter_invocations=1, synthetic=True)
    tracker.record(adapter_invocations=1, synthetic=True)
    with pytest.raises(BudgetExhausted):
        tracker.record(adapter_invocations=1, synthetic=True)


def test_budget_exhaustion_partial_results():
    budget = ComputeBudget(max_adapter_invocations=3, policy_id="tight")
    m = run_synthetic_pipeline_smoke(seed=7, budget=budget)
    statuses = [r["execution_status"] for r in m["runs"]]
    assert any(s == "budget_exhausted" for s in statuses) or any(
        r["errors"] for r in m["runs"]
    )


def test_denominator_includes_all_tasks():
    m = run_synthetic_pipeline_smoke(seed=42)
    for cond, agg in m["aggregates"].items():
        assert agg["n_total"] == m["n_tasks"]
        comp = agg.get("completeness", {})
        assert comp.get("n_expected") == m["n_tasks"]


def test_artifacts_written(tmp_path):
    m = run_synthetic_pipeline_smoke(seed=7, output_dir=str(tmp_path))
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.md").exists()
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["seed"] == 7
    assert data["experiment_id"] == m["experiment_id"]
    summary = (tmp_path / "summary.md").read_text()
    assert "execution status" in summary.lower() or "Status separation" in summary


def test_toy_accounting_is_synthetic():
    caps = ToyAdapter().capabilities()
    assert caps.synthetic_accounting is True
    assert caps.reports_model_calls is False
    pred, usage = ToyAdapter().direct_answer(make_demo_tasks()[0])
    assert usage.synthetic is True
    assert usage.model_calls == 0


@pytest.mark.skipif(not tensorgate_available(), reason="TensorGate not installed")
def test_tensorgate_bundle_validation():
    arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = make_latent_bundle(
        arr, model_id="toy", layer_id="l0", representation_semantics="test"
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
