# Experiment protocol

## Fair comparison

- All conditions receive the same task set and master seed.
- Intervention seeds are derived via `neuralese.core.seeding.stable_seed`
  (SHA-256; independent of `PYTHONHASHSEED`).
- Compute budgets are declared up front and **enforced** during execution.
- Test answers are not injected into model inputs.

## Status separation

| Field | Meaning |
|-------|---------|
| `execution_status` | completed / error / budget_exhausted / skipped |
| `condition_outcome` | met / not_met / inconclusive |
| `hypothesis_result` | supported / not_supported / inconclusive |

A completed run with a negative scientific result remains **completed**.

## Reproducibility

Fixed master seed → identical predictions, intervention records, and aggregates
across processes. Timestamps and run IDs may differ.

## Interpretation

This synthetic benchmark verifies **pipeline mechanics**, not latent reasoning
in real language models.
