# Reproducibility rules

Python 3.11 in `environment.yml` is the canonical runtime for the first scientific runs. Python 3.12 is a compatibility target in CI. Keep `environment.yml` portable and human-maintained; record resolved versions in each run artifact instead of committing a platform-specific full export.

Each run must store its fully resolved YAML or JSON config, algorithm and environment ID, wrapper order, seed, Git commit hash, UTC timestamps, exact Python, PyTorch, Gymnasium, NumPy, and other package versions plus the full platform string, OS/CPU/GPU metadata, deterministic backend settings, and run status. The scaffold's metadata helper writes the common JSON record; a future trainer must add start/end and output manifests.

Seed Python, NumPy, PyTorch, environment resets, and action spaces where supported. Record limitations: exact determinism is not guaranteed for every device, operation, parallel schedule, or library version. Keep CPU as the default and CUDA opt-in.

Evaluation must use a separate seeded environment without exploration noise. Save raw per-step or per-episode CSV/JSONL, plus optional TensorBoard events, for every planned seed. Keep failed and divergent runs. Aggregate in a separate report with the raw run IDs and analysis script/config. Do not commit generated outputs, datasets, checkpoints, secrets, or environment directories.
