# Contributing

1. Work on a feature branch (`feat/*` or `fix/*`); do not commit to `main` directly.
2. Keep core dependencies minimal; optional integrations (TensorGate, real model adapters) stay optional.
3. Do not commit credentials, proprietary datasets, or machine-specific paths.
4. Add tests for new experiment conditions and interventions.
5. Do not overclaim scientific results in docs or commit messages.
6. Run `pytest -q` and `neuralese run synthetic --seed 42` before opening a PR.
