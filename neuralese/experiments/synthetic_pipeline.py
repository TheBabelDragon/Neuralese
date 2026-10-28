"""Synthetic pipeline & causal-intervention smoke test.

NOT evidence that real LMs benefit from latent reasoning.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from neuralese.adapters.toy import ToyAdapter
from neuralese.compute.budget import BudgetExhausted, BudgetTracker, ComputeBudget
from neuralese.core.seeding import task_index_seed
from neuralese.core.types import (
    ConditionOutcome,
    EvaluationCompleteness,
    EvaluationOutcome,
    ExecutionStatus,
    ExperimentCondition,
    HypothesisResult,
    ReasoningRun,
    ReasoningTask,
    ReferenceAnswer,
    RunStatus,
)
from neuralese.interventions.ops import ablate_latent, assert_ids_differ, corrupt_latent, shuffle_latent
from neuralese.metrics.eval import aggregate_outcomes, exact_match
from neuralese.results.manifest import build_manifest, write_json, write_markdown_summary
from neuralese.tensorgate_adapter.bridge import tensorgate_available, tensorgate_status

SYNTHETIC_EXPERIMENT_ID = "synthetic_pipeline_smoke_v2"
DATASET_ID = "synthetic_symbol_slots_v2"
SPLIT_ID = "demo"
MANIFEST_SCHEMA_VERSION = "1.1.0"

CRITERIA = {
    ExperimentCondition.DIRECT: {
        "hypothesis": "H_direct: direct answers achieve accuracy == 1.0 on the synthetic set",
        "min_accuracy": 1.0,
    },
    ExperimentCondition.TEXT_INTERMEDIATE: {
        "hypothesis": "H_text: text intermediate achieves accuracy == 1.0",
        "min_accuracy": 1.0,
    },
    ExperimentCondition.LATENT_INTERMEDIATE: {
        "hypothesis": "H_latent: intact latent recovers accuracy == 1.0",
        "min_accuracy": 1.0,
    },
    ExperimentCondition.LATENT_SHUFFLED: {
        "hypothesis": "H_shuffle: shuffle degrades accuracy to < 0.5 on this task family",
        "max_accuracy": 0.5,
    },
    ExperimentCondition.LATENT_ABLATED: {
        "hypothesis": "H_ablate: ablation degrades accuracy to < 0.5",
        "max_accuracy": 0.5,
    },
    ExperimentCondition.LATENT_NOISE_CONTROL: {
        "hypothesis": "H_noise_control: tiny irrelevant noise preserves accuracy == 1.0",
        "min_accuracy": 1.0,
    },
}


def make_demo_tasks() -> List[ReasoningTask]:
    combos = [
        ("RED", "SQUARE"),
        ("BLUE", "CIRCLE"),
        ("GREEN", "TRIANGLE"),
        ("YELLOW", "HEXAGON"),
        ("RED", "CIRCLE"),
        ("BLUE", "SQUARE"),
        ("GREEN", "HEXAGON"),
        ("YELLOW", "TRIANGLE"),
    ]
    tasks = []
    for i, (color, shape) in enumerate(combos):
        tasks.append(
            ReasoningTask(
                task_id=f"t{i+1:02d}",
                input_text=f"Object is {color} {shape}",
                intermediate_slots={"color": color, "shape": shape},
                metadata={"task_index": i},
            )
        )
    return tasks


def _reference(task: ReasoningTask) -> ReferenceAnswer:
    ans = " ".join(
        str(task.intermediate_slots[k]).upper() for k in sorted(task.intermediate_slots)
    )
    return ReferenceAnswer(task_id=task.task_id, answer=ans, source="synthetic_slots")


def _apply_usage(tracker: BudgetTracker, usage) -> None:
    tracker.record(
        adapter_invocations=usage.adapter_invocations,
        model_calls=usage.model_calls,
        tokens=usage.tokens if usage.tokens is not None else None,
        steps=usage.steps,
        synthetic=usage.synthetic,
    )


def _run_condition(
    condition: ExperimentCondition,
    tasks: List[ReasoningTask],
    adapter: ToyAdapter,
    seed: int,
    budget: ComputeBudget,
) -> ReasoningRun:
    run_id = f"{condition.value}-{uuid.uuid4().hex[:8]}"
    tracker = BudgetTracker(budget)
    tracker.start()
    predictions = []
    outcomes: List[EvaluationOutcome] = []
    latent_ids: List[str] = []
    interventions: List[Dict[str, Any]] = []
    errors: List[str] = []
    execution = ExecutionStatus.COMPLETED

    for idx, task in enumerate(tasks):
        ref = _reference(task)
        try:
            if condition == ExperimentCondition.DIRECT:
                pred, usage = adapter.direct_answer(task)
                _apply_usage(tracker, usage)
            elif condition == ExperimentCondition.TEXT_INTERMEDIATE:
                _text, pred, usage = adapter.text_intermediate(task)
                _apply_usage(tracker, usage)
            elif condition == ExperimentCondition.LATENT_INTERMEDIATE:
                latent, u1 = adapter.produce_latent(task)
                _apply_usage(tracker, u1)
                lid = getattr(latent, "latent_id", "")
                if lid:
                    latent_ids.append(lid)
                pred, u2 = adapter.predict_from_latent(task, latent)
                _apply_usage(tracker, u2)
            elif condition == ExperimentCondition.LATENT_SHUFFLED:
                latent, u1 = adapter.produce_latent(task)
                _apply_usage(tracker, u1)
                iseed = task_index_seed(seed, idx, "shuffle")
                mod, rec = shuffle_latent(latent, seed=iseed)
                rec["derived_seed"] = iseed
                rec["task_index"] = idx
                interventions.append(rec)
                if getattr(latent, "latent_id", None):
                    assert_ids_differ(latent, mod)
                    latent_ids.append(getattr(mod, "latent_id", "") or "")
                pred, u2 = adapter.predict_from_latent(task, mod)
                _apply_usage(tracker, u2)
            elif condition == ExperimentCondition.LATENT_ABLATED:
                latent, u1 = adapter.produce_latent(task)
                _apply_usage(tracker, u1)
                mod, rec = ablate_latent(latent, mode="zero")
                rec["task_index"] = idx
                interventions.append(rec)
                if getattr(latent, "latent_id", None):
                    assert_ids_differ(latent, mod)
                    latent_ids.append(getattr(mod, "latent_id", "") or "")
                pred, u2 = adapter.predict_from_latent(task, mod)
                _apply_usage(tracker, u2)
            elif condition == ExperimentCondition.LATENT_NOISE_CONTROL:
                latent, u1 = adapter.produce_latent(task)
                _apply_usage(tracker, u1)
                iseed = task_index_seed(seed, idx, "noise_control")
                mod, rec = corrupt_latent(latent, seed=iseed, scale=1e-6)
                rec["derived_seed"] = iseed
                rec["task_index"] = idx
                rec["role"] = "negative_control"
                interventions.append(rec)
                if getattr(latent, "latent_id", None):
                    assert_ids_differ(latent, mod)
                    latent_ids.append(getattr(mod, "latent_id", "") or "")
                pred, u2 = adapter.predict_from_latent(task, mod)
                _apply_usage(tracker, u2)
            else:
                raise ValueError(f"unknown condition {condition}")
            predictions.append(pred)
            outcomes.append(exact_match(pred, ref))
        except BudgetExhausted as exc:
            execution = ExecutionStatus.BUDGET_EXHAUSTED
            errors.append(f"{task.task_id}: {exc}")
            outcomes.append(
                EvaluationOutcome(
                    task_id=task.task_id,
                    correct=None,
                    status=RunStatus.ERROR,
                    detail=f"budget_exhausted: {exc}",
                )
            )
            break
        except Exception as exc:
            errors.append(f"{task.task_id}: {exc}")
            outcomes.append(
                EvaluationOutcome(
                    task_id=task.task_id,
                    correct=None,
                    status=RunStatus.ERROR,
                    detail=str(exc),
                )
            )

    tracker.stop()

    n_expected = len(tasks)
    n_completed = sum(1 for o in outcomes if o.status in (RunStatus.PASSED, RunStatus.FAILED))
    n_failed = sum(1 for o in outcomes if o.correct is False)
    n_inconclusive = sum(1 for o in outcomes if o.correct is None and o.status != RunStatus.ERROR)
    n_error = sum(1 for o in outcomes if o.status == RunStatus.ERROR)
    completeness = EvaluationCompleteness(
        n_expected=n_expected,
        n_completed=n_completed,
        n_failed=n_failed,
        n_inconclusive=n_inconclusive,
        n_error=n_error,
    )

    agg = aggregate_outcomes(outcomes)
    accuracy = agg.get("accuracy")
    crit = CRITERIA.get(condition, {})
    condition_outcome = ConditionOutcome.INCONCLUSIVE
    hypothesis_result = HypothesisResult.INCONCLUSIVE
    if accuracy is None or completeness.n_completed == 0:
        condition_outcome = ConditionOutcome.INCONCLUSIVE
        hypothesis_result = HypothesisResult.INCONCLUSIVE
    else:
        min_acc = crit.get("min_accuracy")
        max_acc = crit.get("max_accuracy")
        met = True
        if min_acc is not None and accuracy < min_acc:
            met = False
        if max_acc is not None and accuracy >= max_acc:
            met = False
        condition_outcome = ConditionOutcome.MET if met else ConditionOutcome.NOT_MET
        hypothesis_result = (
            HypothesisResult.SUPPORTED if met else HypothesisResult.NOT_SUPPORTED
        )

    if errors and execution == ExecutionStatus.COMPLETED:
        if n_error == n_expected:
            execution = ExecutionStatus.ERROR

    legacy = RunStatus.PASSED
    if execution == ExecutionStatus.ERROR:
        legacy = RunStatus.ERROR
    elif execution == ExecutionStatus.BUDGET_EXHAUSTED:
        legacy = RunStatus.ERROR
    elif condition_outcome == ConditionOutcome.NOT_MET:
        legacy = RunStatus.FAILED

    return ReasoningRun(
        run_id=run_id,
        experiment_id=SYNTHETIC_EXPERIMENT_ID,
        condition=condition,
        seed=seed,
        execution_status=execution,
        condition_outcome=condition_outcome,
        hypothesis_result=hypothesis_result,
        completeness=completeness,
        predictions=predictions,
        outcomes=outcomes,
        latent_ids=[x for x in latent_ids if x],
        interventions=interventions,
        budget=tracker.usage.to_dict(),
        errors=errors,
        extras={"criteria": crit, "accuracy": accuracy, "aggregate": agg},
        status=legacy,
    )


def run_synthetic_pipeline_smoke(
    *,
    seed: int = 42,
    output_dir: Optional[str] = None,
    budget: Optional[ComputeBudget] = None,
) -> Dict[str, Any]:
    from pathlib import Path

    tasks = make_demo_tasks()
    adapter = ToyAdapter()
    if budget is None:
        budget = ComputeBudget(
            max_adapter_invocations=500,
            max_steps=500,
            policy_id="synthetic_demo_v1",
        )
    conditions = [
        ExperimentCondition.DIRECT,
        ExperimentCondition.TEXT_INTERMEDIATE,
        ExperimentCondition.LATENT_INTERMEDIATE,
        ExperimentCondition.LATENT_SHUFFLED,
        ExperimentCondition.LATENT_ABLATED,
        ExperimentCondition.LATENT_NOISE_CONTROL,
    ]

    runs: List[ReasoningRun] = []
    aggregates: Dict[str, Any] = {}
    for cond in conditions:
        run = _run_condition(cond, tasks, adapter, seed, budget)
        runs.append(run)
        aggregates[cond.value] = {
            **aggregate_outcomes(run.outcomes),
            "execution_status": run.execution_status.value,
            "condition_outcome": run.condition_outcome.value,
            "hypothesis_result": run.hypothesis_result.value,
            "completeness": run.completeness.to_dict(),
            "criteria": run.extras.get("criteria", {}),
        }

    run_dicts = []
    for r in runs:
        run_dicts.append(
            {
                "run_id": r.run_id,
                "condition": r.condition.value,
                "execution_status": r.execution_status.value,
                "condition_outcome": r.condition_outcome.value,
                "hypothesis_result": r.hypothesis_result.value,
                "completeness": r.completeness.to_dict(),
                "seed": r.seed,
                "predictions": [
                    {"task_id": p.task_id, "answer": p.answer, "confidence": p.confidence}
                    for p in r.predictions
                ],
                "outcomes": [
                    {
                        "task_id": o.task_id,
                        "correct": o.correct,
                        "status": o.status.value,
                        "detail": o.detail,
                    }
                    for o in r.outcomes
                ],
                "latent_ids": r.latent_ids,
                "interventions": r.interventions,
                "budget": r.budget,
                "errors": r.errors,
                "criteria": r.extras.get("criteria", {}),
            }
        )

    manifest = build_manifest(
        experiment_id=SYNTHETIC_EXPERIMENT_ID,
        seed=seed,
        adapter_identity={
            **adapter.identity(),
            "capabilities": adapter.capabilities().to_dict(),
        },
        tensorgate_status=tensorgate_status(),
        conditions=[c.value for c in conditions],
        dataset_id=DATASET_ID,
        split_id=SPLIT_ID,
        budget=budget.to_dict(),
        runs=run_dicts,
        aggregates=aggregates,
        hypotheses=[c["hypothesis"] for c in CRITERIA.values()],
        limitations=[
            "Synthetic symbol encoder only — not a real language model.",
            "Does not demonstrate latent reasoning benefits in foundation models.",
            "Pipeline / causal-intervention smoke test only.",
            "Toy adapter accounting is synthetic (model_calls=0).",
            f"TensorGate available: {tensorgate_available()}",
        ],
    )
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["master_seed"] = seed
    manifest["task_ids"] = [t.task_id for t in tasks]
    manifest["n_tasks"] = len(tasks)
    manifest["seed_algorithm"] = (
        "SHA-256(master|role|task_index) -> uint32 mod 2^31-1; see neuralese.core.seeding"
    )

    if output_dir:
        out = Path(output_dir)
        write_json(out / "manifest.json", manifest)
        write_markdown_summary(out / "summary.md", manifest)

    return manifest


def normalize_manifest_for_compare(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Strip intentionally non-deterministic fields for cross-process comparison."""
    out = {
        "experiment_id": manifest.get("experiment_id"),
        "master_seed": manifest.get("master_seed", manifest.get("seed")),
        "dataset_id": manifest.get("dataset_id"),
        "split_id": manifest.get("split_id"),
        "n_tasks": manifest.get("n_tasks"),
        "task_ids": manifest.get("task_ids"),
        "aggregates": {},
        "predictions_by_condition": {},
        "intervention_seeds": {},
    }
    for cond, agg in (manifest.get("aggregates") or {}).items():
        out["aggregates"][cond] = {
            k: agg[k]
            for k in (
                "n_total",
                "n_correct",
                "n_wrong",
                "n_inconclusive",
                "n_error",
                "accuracy",
                "execution_status",
                "condition_outcome",
                "hypothesis_result",
            )
            if k in agg
        }
    for run in manifest.get("runs") or []:
        cond = run["condition"]
        out["predictions_by_condition"][cond] = [
            {"task_id": p["task_id"], "answer": p["answer"]} for p in run.get("predictions", [])
        ]
        seeds = [
            i.get("derived_seed")
            for i in run.get("interventions", [])
            if i.get("derived_seed") is not None
        ]
        if seeds:
            out["intervention_seeds"][cond] = seeds
    return out
