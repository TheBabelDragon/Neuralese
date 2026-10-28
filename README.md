# Neuralese

**Experimental infrastructure for reproducible evaluation of latent-state reasoning, representation interventions, and adaptive computation.**

[TensorGate](https://github.com/TheBabelDragon/TensorGate) is the representation and validation layer.
**Neuralese** is the experiment runner, model-adapter layer, and benchmark suite.

> This repository does **not** claim that real language models benefit from latent-state reasoning. The included synthetic experiment is a **pipeline and causal-intervention smoke test only**.

## Install

```bash
pip install -e ".[dev]"
pip install tensorgate  # optional but recommended
```

## Quick start

```bash
neuralese list-experiments
neuralese tensorgate-status
neuralese run synthetic --seed 42 --out artifacts/synthetic_run
neuralese summarize artifacts/synthetic_run/manifest.json
```

## License

MIT
