# Changelog

## Unreleased

- Bootstrap the research repository with documentation, configuration, algorithm interfaces, CLI inspection commands, and fast scaffold tests. No training algorithm is implemented.

- Add typed action, transition, rollout, update, and metric contracts; deterministic seeding; continuous-action transforms; replay with explicit truncation masks; tiny and Pendulum environment creation; no-learning trainer/evaluator harness; and fast smoke tests. Named learning algorithms remain planned.

- Implement and verify the project-owned four-network DDPG baseline with replay TD regression, deterministic policy gradients, collection-only exploration, soft target updates, checkpointing, local metrics, and CPU smoke coverage. A single smoke run is not scientific evidence.

- Implement the experimental target-free three-network shared-error core with sequential and simultaneous schedules, explicit stop-gradient paths, optional actor/residual diagnostics, checkpointing, strict configuration validation, and deterministic and Pendulum smoke coverage. Agreement penalties, beta regularization, guarded updates, and convergence claims remain out of scope.
