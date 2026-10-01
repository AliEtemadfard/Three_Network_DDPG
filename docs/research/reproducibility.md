# Reproducibility rules

Python 3.11 in `environment.yml` is the canonical runtime for the first scientific runs. Python 3.12 is a compatibility target in CI. Keep `environment.yml` portable and human-maintained; record resolved versions in each run artifact instead of committing a platform-specific full export.

Each run stores its fully resolved YAML or JSON config, configuration hash, stable run and variant IDs, algorithm and environment ID, wrapper order, training seed, separate evaluation seeds, Git commit hash, UTC start/end timestamps, exact Python, PyTorch, Gymnasium, NumPy, and other package versions plus the full platform string, OS/CPU/GPU metadata, deterministic backend settings, and run status.

Seed Python, NumPy, PyTorch, environment resets, and action spaces where supported. Record limitations: exact determinism is not guaranteed for every device, operation, parallel schedule, or library version. Keep CPU as the default and CUDA opt-in.

Evaluation must use a separate seeded environment without exploration noise. Save raw per-step or per-episode CSV/JSONL, plus optional TensorBoard events, for every planned seed. Keep failed and divergent runs. Aggregate in a separate report with the raw run IDs and analysis script/config. Do not commit generated outputs, datasets, checkpoints, secrets, or environment directories.

The core run-directory helper writes a resolved config and metadata with the active Git commit, selected run seed, package versions, UTC start time, and CPU/GPU and OS information. Manifest execution additionally freezes the source manifest and run matrix, writes append-safe metrics during training, records an atomic terminal marker, and preserves immutable attempts for audit and resume. The DDPG and three-network experiment entry points create run directories and write resolved configuration, metadata, JSONL metrics, summaries, and checkpoints. Set `clip_actions` and `bootstrap_on_truncation` explicitly in the experiment config; the replay buffer and trainer reject conflicting truncation policies. Use a separate environment instance for evaluation.

`run` selects only never-started run IDs. `resume` selects incomplete, interrupted, and failed IDs and never overwrites an old attempt. Divergent and non-finite outcomes are retained as terminal evidence. Aggregation includes all planned statuses and failure reasons while excluding unsuccessful values from numerical summaries. A changed manifest or Git commit cannot silently reuse an existing study directory.
