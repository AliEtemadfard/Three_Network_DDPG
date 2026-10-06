# Changelog

## Unreleased

- Bootstrap the research repository with documentation, configuration, algorithm interfaces, CLI inspection commands, and fast scaffold tests. No training algorithm is implemented.

- Establish Python 3.11 as the canonical Conda runtime, add an environment smoke check and Python 3.12 CI compatibility job, and record exact runtime and platform versions in run metadata.

- Add typed action, transition, rollout, update, and metric contracts; deterministic seeding; continuous-action transforms; replay with explicit truncation masks; tiny and Pendulum environment creation; no-learning trainer/evaluator harness; and fast smoke tests. Named learning algorithms remain planned.

- Implement and verify the project-owned four-network DDPG baseline with replay TD regression, deterministic policy gradients, collection-only exploration, soft target updates, checkpointing, local metrics, and CPU smoke coverage. A single smoke run is not scientific evidence.

- Implement the experimental target-free three-network shared-error core with sequential and simultaneous schedules, explicit stop-gradient paths, optional actor/residual diagnostics, checkpointing, strict configuration validation, and deterministic and Pendulum smoke coverage. Agreement penalties, beta regularization, guarded updates, and convergence claims remain out of scope.
- Add controlled current and next critic agreement, experimental beta variants, same-snapshot simultaneous updates, recomputing sequential updates, scoped guarded backtracking with full optimizer restoration, independent validation monitoring, and interval-controlled critic and actor gradient diagnostics. These mechanisms remain experimental and establish no convergence claim.

- Reject the undefined l2_y_z beta mode, honor DDPG exploration mode, centralize auto device resolution, enforce semantic checkpoint compatibility, and report guarded backtracking metrics for the last executed proposal.

- Add the experimental `ddpg_no_target_actor` baseline with detached online-actor bootstrapping through a target critic, target-critic-only Polyak updates, distinct configuration and checkpoint identity, and deterministic and Pendulum smoke coverage. Stability remains unestablished pending multi-seed studies.

- Add versioned experiment manifests, stable run identities, paired training and fixed evaluation seeds, crash-safe attempts and resume, explicit completion/failure states, failure-aware bootstrap aggregation, per-seed SVG plots, bounded smoke infrastructure, and PowerShell/Bash operating instructions. No scientific result is claimed.

- Prepare a manually launched, three-stage Pendulum-v1 development pilot with separated manifests, paired development seeds, held-out confirmatory seeds, stage gates, workload accounting, focused tests, and a PowerShell execution guide. No Pendulum training is run by this preparation change.

- Make experiment SVG plots quantitatively readable with labelled numeric axes, observation markers, explicit seed and confidence-interval status, failure counts, and a shared-scale evaluation-return comparison figure. Mechanism metrics remain per-variant when they are undefined for baseline algorithms.

- Add an opt-in, sequential-only `separate_td` three-network critic objective that regresses both current predictions against a detached Critic Two bootstrap, preserves the original shared-error default, logs both residual systems, and includes a bounded manual Pendulum screening manifest. No screening run is executed by this change.
