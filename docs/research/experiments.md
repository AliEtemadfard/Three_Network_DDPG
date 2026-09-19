# Experiment protocol

## Sequence

1. Verify standard DDPG on a small deterministic continuous task and Pendulum-v1 before implementing the proposal.
2. Test the minimal shared-error method with no agreement or beta term. Check gradients, terminal masks, action scaling, and absence of target networks.
3. Add current and next agreement penalties separately, then together.
4. Compare beta modes, update schedules, actor state sources, critic initialization, actor delay, alpha values, and guarded controls.
5. Run at least ten planned seeds on Pendulum-v1, then one harder continuous-control environment after stability is established.

Use the same environment version, wrappers, normalization, network capacity, replay settings, exploration noise, evaluation cadence, and training budget for direct comparisons. Report exceptions and compute cost. DDPG without a target actor is a diagnostic ablation; TD3 is a strong direct reference. See [baseline policy](baselines.md).

## Diagnostics

Record shared TD error before and after each phase, current and next agreement, each component loss, mean and maximum absolute Q, actor–residual gradient inner product, both critic alignment inner products, accepted/rejected/backtracked steps, actor objective, episode return and success, wall time, parameter count, and memory. Check descent on a separate replay batch when guarded updates use an acceptance batch. Low shared error alone does not establish good control.

Report all seeds, median or mean with uncertainty, and catastrophic divergence. Do not choose a variant based only on one run. Any runtime or memory advantage must be measured. Keep raw results under `runs/` and aggregate reports under `reports/`.
