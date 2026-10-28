"""Synthetic pipeline & causal-intervention smoke test.

NOT evidence that real LMs benefit from latent reasoning.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from neuralese.adapters.toy import ToyAdapter
from neuralese.compute.budget import BudgetTracker, ComputeBudget
from neuralese.core.types import (
    ExperimentCondition,
    ReasoningRun,
    ReasoningTask,
    ReferenceAnswer,
    RunStatus,
)
from neuralese.interventions.ops import ablate_latent, assert_ids_differ, shuffle_latent
from neuralese.metrics.eval import aggregate_outcomes, exact_match
from neuralese.results.manifest import build_manifest, write_json, write_markdown_summary
from neuralese.tensorgate_adapter.bridge import tensorgate_available, tensorgate_status

SYNTHETIC_EXPERIMENT_ID = "synthetic_pipeline_smoke_v1"
DATASET_ID = "synthetic_symbol_slots_v1"
SPLIT_ID = "demo"


def make_demo_tasks() -> List[ReasoningTask]:
    return [
        ReasoningTask(
            task_id="t1",
            input_text="Object is RED SQUARE",
            intermediate_slots={"color": "RED", "shape": "SQUARE"},
        ),
        ReasoningTask(
            task_id="t2",
            input_text="Object is BLUE CIRCLE",
            intermediate_slots={"color": "BLUE", "shape": "CIRCLE"},
        ),
        ReasoningTask(
            task_id="t3",
            input_text="Object is GREEN TRIANGLE",
            intermediate_slots={"color": "GREEN", "shape": "TRIANGLE"},
        ),
    ]


def _reference(task: ReasoningTask) -> ReferenceAnswer:
    ans = " ".join(
        str(task.intermediate_slots[k]).upper() for k in sorted(task.intermediate_slots)
    )
    return ReferenceAnswer(task_id=task.task_id, answer=ans, source="synthetic_slots")


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
    outcomes = []
    latent_ids: List[str] = []
    interventions: List[Dict[str, Any]] = []
    errors: List[str] = []

    for task in tasks:
        ref = _reference(task)
        try:
            if condition == ExperimentCondition.DIRECT:
                tracker.record_call()
                pred = adapter.direct_answer(task)
            elif condition == ExperimentCondition.TEXT_INTERMEDIATE:
                tracker.record_call()
                _text, pred = adapter.text_intermediate(task)
            elif condition == ExperimentCondition.LATENT_INTERMEDIATE:
                tracker.record_call()
                latent = adapter.produce_latent(task)
                lid = getattr(latent, "latent_id", "")
                if lid:
                    latent_ids.append(lid)
                pred = adapter.predict_from_latent(task, latent)
            elif condition == ExperimentCondition.LATENT_SHUFFLED:
                tracker.record_call()
                latent = adapter.produce_latent(task)
                mod, rec = shuffle_latent(latent, seed=seed + hash(task.task_id) % 1000)
                interventions.append(rec)
                if getattr(latent, "latent_id", None):
                    assert_ids_differ(latent, mod)
                    latent_ids.append(getattr(mod, "latent_id", ""))
                pred = adapter.predict_from_latent(task, mod)
            elif condition == ExperimentCondition.LATENT_ABLATED:
                tracker.record_call()
                latent = adapter.produce_latent(task)
                mod, rec = ablate_latent(latent, mode="zero")
                interventions.append(rec)
                if getattr(latent, "latent_id", None):
                    assert_ids_differ(latent, mod)
                    latent_ids.append(getattr(mod, "latent_id", ""))
                pred = adapter.predict_from_latent(task, mod)
            else:
                raise ValueError(f"unknown condition {condition}")
            predictions.append(pred)
            outcomes.append(exact_match(pred, ref))
        except Exception as exc:
            errors.append(f"{task.task_id}: {exc}")
            from neuralese.core.types import EvaluationOutcome as EO

            outcomes.append(
                EO(
                    task_id=task.task_id,
                    correct=None,
                    status=RunStatus.ERROR,
                    detail=str(exc),
                )
            )

    tracker.stop()
    status = RunStatus.PASSED
    if errors:
        status = RunStatus.ERROR
    elif any(o.correct is False for o in outcomes) and condition in (
        ExperimentCondition.LATENT_SHUFFLED,
        ExperimentCondition.LATENT_ABLATED,
    ):
        status = RunStatus.PASSED
    elif any(o.correct is False for o in outcomes):
        status = RunStatus.FAILED

    return ReasoningRun(
        run_id=run_id,
        experiment_id=SYNTHETIC_EXPERIMENT_ID,
        condition=condition,
        status=status,
        seed=seed,
        predictions=predictions,
        outcomes=outcomes,
        latent_ids=[x for x in latent_ids if x],
        interventions=interventions,
        budget=tracker.usage.to_dict(),
        errors=errors,
    )


def run_synthetic_pipeline_smoke(
    *,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    from pathlib import Path

    tasks = make_demo_tasks()
    adapter = ToyAdapter()
    budget = ComputeBudget(max_model_calls=100, max_steps=100)
    conditions = [
        ExperimentCondition.DIRECT,
        ExperimentCondition.TEXT_INTERMEDIATE,
        ExperimentCondition.LATENT_INTERMEDIATE,
        ExperimentCondition.LATENT_SHUFFLED,
        ExperimentCondition.LATENT_ABLATED,
    ]

    runs: List[ReasoningRun] = []
    aggregates: Dict[str, Any] = {}
    for cond in conditions:
        run = _run_condition(cond, tasks, adapter, seed, budget)
        runs.append(run)
        aggregates[cond.value] = aggregate_outcomes(run.outcomes)

    run_dicts = []
    for r in runs:
        run_dicts.append(
            {
                "run_id": r.run_id,
                "condition": r.condition.value,
                "status": r.status.value,
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
            }
        )

    manifest = build_manifest(
        experiment_id=SYNTHETIC_EXPERIMENT_ID,
        seed=seed,
        adapter_identity=adapter.identity(),
        tensorgate_status=tensorgate_status(),
        conditions=[c.value for c in conditions],
        dataset_id=DATASET_ID,
        split_id=SPLIT_ID,
        budget=budget.to_dict(),
        runs=run_dicts,
        aggregates=aggregates,
        hypotheses=[
            "H1: Intact latent intermediate recovers correct answers on this synthetic task.",
            "H2: Shuffling or ablating the latent degrades accuracy when the task requires the intermediate.",
            "H3: Direct and text baselines remain solvable under the same budget.",
        ],
        limitations=[
            "Synthetic symbol encoder only — not a real language model.",
            "Does not demonstrate latent reasoning benefits in foundation models.",
            "Pipeline / causal-intervention smoke test only.",
            f"TensorGate available: {tensorgate_available()}",
        ],
    )

    if output_dir:
        out = Path(output_dir)
        write_json(out / "manifest.json", manifest)
        write_markdown_summary(out / "summary.md", manifest)

    return manifest
