"""Task metrics with explicit denominators and visible failures."""

from __future__ import annotations

from typing import Any, Dict, List

from neuralese.core.types import EvaluationOutcome, Prediction, ReferenceAnswer, RunStatus


def exact_match(prediction: Prediction, reference: ReferenceAnswer) -> EvaluationOutcome:
    pred = str(prediction.answer).strip().upper()
    ref = str(reference.answer).strip().upper()
    if not pred or not ref:
        return EvaluationOutcome(
            task_id=prediction.task_id,
            correct=None,
            status=RunStatus.INCONCLUSIVE,
            detail="empty prediction or reference",
        )
    ok = pred == ref
    return EvaluationOutcome(
        task_id=prediction.task_id,
        correct=ok,
        status=RunStatus.PASSED if ok else RunStatus.FAILED,
        detail="exact_match",
        metrics={"exact_match": ok},
    )


def aggregate_outcomes(outcomes: List[EvaluationOutcome]) -> Dict[str, Any]:
    n = len(outcomes)
    n_correct = sum(1 for o in outcomes if o.correct is True)
    n_wrong = sum(1 for o in outcomes if o.correct is False)
    n_inconclusive = sum(1 for o in outcomes if o.correct is None)
    n_error = sum(1 for o in outcomes if o.status == RunStatus.ERROR)
    accuracy = n_correct / n if n else None
    return {
        "n_total": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_inconclusive": n_inconclusive,
        "n_error": n_error,
        "accuracy": accuracy,
        "accuracy_note": "accuracy = n_correct / n_total (failures included in denominator)",
    }
