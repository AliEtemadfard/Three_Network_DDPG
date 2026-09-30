# Contributing

Read [the research problem](docs/research/problem.md), [notation](docs/research/notation.md), and [architecture](docs/research/architecture.md) before changing algorithm code. The original specification is preserved in `docs/research/source/`.

Use the Python 3.11 Conda environment from `environment.yml`; run `python scripts/environment_smoke.py` after creating or updating it. Use a focused feature branch from `main`, a pull request, and at least one reviewer. Use conventional commits such as `docs: clarify stop-gradient matrix` or `test: cover terminal masking`. Keep experimental choices in configuration, add focused tests for gradient or update-order changes, and update the relevant docs and `CHANGELOG.md`. Run `ruff check .`, `ruff format --check .`, and `python -m pytest`; inspect `git diff --check` before committing. Never commit data, runs, checkpoints, caches, secrets, or environment directories.

Do not copy baseline source code until its URL, commit or version, license, and local modifications are recorded in `THIRD_PARTY_NOTICES.md`. DDPG verification comes first. See [baseline policy](docs/research/baselines.md).
