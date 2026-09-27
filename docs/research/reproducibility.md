# Reproducibility rules

Each run must store its fully resolved YAML or JSON config, algorithm and environment ID, wrapper order, seed, Git commit hash, UTC timestamps, Python/PyTorch/Gymnasium and other package versions, OS/CPU/GPU metadata, deterministic backend settings, and run status. The scaffold's metadata helper writes the common JSON record; a future trainer must add start/end and output manifests.

Seed Python, NumPy, PyTorch, environment resets, and action spaces where supported. Record limitations: exact determinism is not guaranteed for every device, operation, parallel schedule, or library version. Keep CPU as the default and CUDA opt-in.

Evaluation must use a separate seeded environment without exploration noise. Save raw per-step or per-episode CSV/JSONL, plus optional TensorBoard events, for every planned seed. Keep failed and divergent runs. Aggregate in a separate report with the raw run IDs and analysis script/config. Do not commit generated outputs, datasets, checkpoints, secrets, or environment directories.

The core run-directory helper now writes a resolved config and metadata with the active Git commit, selected run seed, package versions, UTC start time, and CPU/GPU and OS information. The trainer shell remains tracking-agnostic. The DDPG and three-network experiment entry points create run directories and write resolved configuration, metadata, JSONL metrics, summaries, and checkpoints. Set `clip_actions` and `bootstrap_on_truncation` explicitly in the experiment config; the replay buffer and trainer reject conflicting truncation policies. Use a separate environment instance for evaluation.
