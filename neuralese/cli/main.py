"""Neuralese CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neuralese._version import __software_id__, __version__
from neuralese.experiments.synthetic_pipeline import (
    SYNTHETIC_EXPERIMENT_ID,
    run_synthetic_pipeline_smoke,
)
from neuralese.tensorgate_adapter.bridge import tensorgate_status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralese",
        description="Neuralese — latent-state reasoning evaluation framework",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print version")
    sub.add_parser("list-experiments", help="List available experiments")
    sub.add_parser("tensorgate-status", help="Show TensorGate integration status")

    val = sub.add_parser("validate-config", help="Validate environment/config")
    val.add_argument("--require-tensorgate", action="store_true")

    run_p = sub.add_parser("run", help="Run an experiment")
    run_p.add_argument(
        "experiment",
        choices=[SYNTHETIC_EXPERIMENT_ID, "synthetic"],
        help="Experiment id",
    )
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--out", type=Path, default=Path("artifacts/synthetic_run"))

    sum_p = sub.add_parser("summarize", help="Print summary from a manifest JSON")
    sum_p.add_argument("manifest", type=Path)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "version":
        print(json.dumps({"version": __version__, "software_id": __software_id__}, indent=2))
        return 0

    if args.command == "list-experiments":
        print(
            json.dumps(
                [
                    {
                        "id": SYNTHETIC_EXPERIMENT_ID,
                        "label": "Synthetic pipeline & causal-intervention smoke test",
                        "claim": "Validates harness only — not real LM latent reasoning",
                    }
                ],
                indent=2,
            )
        )
        return 0

    if args.command == "tensorgate-status":
        print(json.dumps(tensorgate_status(), indent=2))
        return 0

    if args.command == "validate-config":
        st = tensorgate_status()
        ok = True
        if args.require_tensorgate and not st.get("available"):
            ok = False
        print(json.dumps({"ok": ok, "tensorgate": st}, indent=2))
        return 0 if ok else 2

    if args.command == "run":
        try:
            manifest = run_synthetic_pipeline_smoke(seed=args.seed, output_dir=str(args.out))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "ok": True,
                    "experiment_id": manifest["experiment_id"],
                    "seed": manifest["seed"],
                    "out": str(args.out),
                    "aggregates": manifest["aggregates"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "summarize":
        data = json.loads(args.manifest.read_text())
        print(json.dumps(data.get("aggregates", {}), indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
