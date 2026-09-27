# Minimal shared-error implementation

## Implemented scope

The experimental `three_network_ddpg` implementation contains one bounded
deterministic actor, Critic One, and Critic Two. The critics are initialized
independently and use separate Adam optimizers. The main variant has no target
actor, target critic, soft update, copied critic, or delayed parameter path.

For every replay batch, with `a_next = actor(s_next)` detached during critic
updates:

- `Y = critic_one(s, a)`
- `Y1 = critic_one(s_next, a_next)`
- `Z1 = critic_two(s, a)`
- `Z = critic_two(s_next, a_next)`
- `delta = r + gamma * bootstrap_mask * Z - Y`
- `shared_squared_error = 0.5 * mean(delta ** 2)`

Critic Two differentiates this loss through `Z` while treating `Y` as
fixed. Critic One differentiates it through `Y` while treating `Z` as
fixed. The actor minimizes
`-mean(critic_two(actor_state, actor(actor_state)))`; `actor_state` defaults
to `s_next` and can be set to `s`. Both critics are frozen during that
forward and backward pass, while the path from Critic Two's action input to the
actor remains connected.

## Schedules

`sequential` applies Critic Two, Critic One, then actor updates. Critic One
recomputes Critic Two's bootstrap value after the Critic Two step. The actor
uses the updated Critic Two. This is the concrete interpretation of the
specification's recommended sequential pseudocode.

`simultaneous` computes and stores all three parameter-gradient sets from one
pre-update parameter snapshot. No optimizer steps before all gradient sets are
stored. Optimizer steps then apply the stored Critic Two, Critic One, and actor
gradients.

## Diagnostics

Every learner update logs the signed and absolute residual means, the shared
squared error, losses, gradient norms, actor objective, and mean, standard
deviation, mean absolute value, and maximum absolute value for `Y`, `Y1`,
`Z1`, and `Z`. It also logs aggregate Q magnitude. The trainer separately
logs episode return and length.

When `diagnostic_frequency` is a positive integer, the implementation measures
the inner product and cosine between the actor-loss gradient and the shared
error gradient with respect to actor parameters. It uses separate
`torch.autograd.grad` calls and never writes these diagnostic gradients into
optimizer state. Set the field to `null` to disable this diagnostic.

## Deliberate deviations and unresolved choices

- The source board expression omitted terminal handling. The implementation
  multiplies `Z` by the replay buffer's explicit bootstrap mask. Terminated and
  truncated flags remain separate; `bootstrap_on_truncation` chooses the
  truncation policy.
- Agreement penalties are absent. `Y1` and `Z1` are diagnostic outputs only.
  Nonzero alpha values or an agreement mode other than `none` are rejected.
- Beta regularization is absent. Nonzero coefficients and all beta modes other
  than `none` are rejected.
- Guarded and legacy `alternating` schedule labels are rejected by this
  implementation. Only the precisely defined `sequential` and `simultaneous`
  schedules execute.
- Critics must use independent initialization. Identical initialization remains
  a future ablation.
- The actor updates on every learner step. Delayed actor updates remain a future
  ablation.
- Adam is the only implemented optimizer. Gradient clipping remains optional
  and disabled by default.
- The loss uses a batch mean and the specified factor of one half. This records
  the mean-versus-sum choice left open by the reconstruction.
- Metrics are measured on the update batch; this does not establish behavior on
  the replay distribution or on unseen transitions.
- Gaussian exploration is used only for collection. Evaluation is deterministic.
- No agreement, beta, guard, target-network, convergence, optimality, monotonic
  return, efficiency, or novelty claim follows from this implementation.

This core establishes executable gradient paths and diagnostics. Scientific
comparison still requires planned multi-seed experiments and theory work.
