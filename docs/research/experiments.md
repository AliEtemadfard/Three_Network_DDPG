# Experiment protocol

## Sequence

1. Maintain the verified standard DDPG baseline on the tiny deterministic task and Pendulum-v1.
2. Compare `ddpg_no_target_actor` with standard DDPG while holding network, replay, optimizer, noise, evaluation, seed, and budget settings fixed.
3. Keep the shared-error-only three-network setting (agreement_mode=none, beta=none) as the reference proposal.
4. Compare current and next agreement separately before enabling both.
5. Compare beta modes and schedules against the shared-error reference while holding all other settings fixed.
6. Compare guard scopes and report acceptance behavior before using a guarded variant in a performance study.
7. Run at least ten planned seeds on Pendulum-v1, then one harder continuous-control environment after stability is established.

Use the same environment version, wrappers, normalization, network capacity, replay settings, exploration noise, evaluation cadence, and training budget for direct comparisons. Report exceptions and compute cost. DDPG and TD3 are the direct baselines described in [baseline policy](baselines.md).

## Controlled ablation matrix

Change one axis at a time unless an interaction is preregistered:

- agreement: none, current, next, both;
- beta: none, experimental linear_y, l2_y;
- schedule: sequential, simultaneous, guarded;
- guarded scope: joint, critics, actor;
- actor input: next, current.

l2_y_z is not an available scientific variant. Symmetric Z regularization would define a new advisor-approved, separately named objective. Start beta and agreement coefficients from zero references and record the tested values. Do not infer that a quadratic beta variant fixes the linear term. Keep critic initialization independent and actor delay at one until separately implemented and tested.

## Guard protocol

The update batch controls acceptance. tolerance, backtracking_factor, maximum_retries, and minimum_learning_rate must be recorded. When independent validation monitoring is enabled, sample a second replay batch and log its shared-error change without using it to accept or reject the proposal. Validation-batch guard metrics remain monitoring-only.

Report accepted and rejected counts, retry distribution, effective candidate learning rate, shared error before and after the candidate, committed error after restoration or acceptance, and validation change. For partial scopes, also report which unguarded phase remained committed.

## Diagnostics

Record shared TD error, current and next agreement, component losses, Y/Y1/Z1/Z statistics, Q magnitude, actor objective, gradient norms, episode return and length, wall time, parameter count, and memory. At the configured diagnostic interval, record both critics' objective/shared-error gradient inner product and cosine plus the actor/shared-error interaction.

Low shared error, agreement, or acceptance rate alone does not establish good control. Report all seeds, central tendency with uncertainty, and catastrophic divergence. Do not select a variant from one run. Keep raw results under runs/ and aggregate reports under reports/.

Smoke tests verify execution and invariants; they are not scientific evidence. In particular, the target-actor ablation must not be described as stable before controlled multi-seed results exist. No variant should advance from smoke output alone.

## Executable manifest protocol

Version 2 manifests pair the same ordered training seeds across every variant and use a separate fixed evaluation-seed set. Stable variant and run hashes derive from the resolved configuration and seed identity. Validation rejects planned algorithms and the undefined `l2_y_z` objective before any run directory is created. The supplied development and confirmatory manifests include DDPG, DDPG without a target actor, shared-error-only, current-agreement, next-agreement, both-agreements, and guarded variants. TD3 remains excluded until it is implemented.

`experiment validate` composes every configuration. `experiment plan` prints the complete matrix without writing artifacts. `experiment run` is run-only-missing, while `experiment resume` creates a new immutable attempt for incomplete, interrupted, or failed runs. Reuse is refused when the frozen manifest hash, resolved configuration hash, or Git commit differs unless `--restart` explicitly archives the incompatible study.

Terminal states are `completed`, `interrupted`, `failed`, `divergent`, and `non_finite`. Aggregation includes every planned run and its failure reason, but computes numerical summaries only from completed runs with finite metrics. Reports include counts, completion rate, mean, sample standard deviation, median, IQR, bootstrap confidence intervals, per-seed curves, last-k return, and normalized learning-curve area. A confidence interval is omitted when fewer than two completed seeds contribute. Curves use the union of environment-step coordinates, interpolate only within each run's observed range, and never extrapolate.

The guarded variant's independent validation batch is monitoring-only. It may be logged and aggregated, but it never controls acceptance or rejection.

## Prepared Pendulum development pilot

`experiments/manifests/pendulum_pilot_01_stage_a.yaml`, `stage_b.yaml`, and `stage_c.yaml` are separate manually launched Pendulum-v1 pilot stages. They share development seeds and fixed evaluation seeds while preserving the confirmatory seed set. The frozen plan records workload and stage gates in [Pendulum Pilot 01](../manual_pendulum_pilot_01.md). Preparation and manifest planning do not constitute execution or scientific evidence.
