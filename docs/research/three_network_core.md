# Three-network controlled implementation

## Fixed method contract

The experimental three_network_ddpg implementation contains one bounded deterministic actor, Critic One, and Critic Two. The critics start independently and use separate Adam optimizers. No target actor, target critic, soft update, copied critic, or delayed parameter path exists in this variant.

For every replay batch, with a_next = actor(s_next) detached during critic updates:

- Y = critic_one(s, a)
- Y1 = critic_one(s_next, a_next)
- Z1 = critic_two(s, a)
- Z = critic_two(s_next, a_next)
- delta = r + gamma * bootstrap_mask * Z - Y
- shared_squared_error = 0.5 * mean(delta ** 2)
- delta_prime = r + gamma * bootstrap_mask * Z - Z1
- q2_td_squared_error = 0.5 * mean(delta_prime ** 2)

Critic Two differentiates through Z while treating Y as fixed. Critic One differentiates through Y while treating Z as fixed. The actor minimizes -mean(critic_two(actor_state, actor(actor_state))); actor_state defaults to s_next and can be set to s. Both critics are frozen during the actor backward pass, while the action to actor path remains connected.

That paragraph describes the default `shared_error` critic objective. It remains unchanged. The
opt-in `separate_td` objective is an experimental reconstruction with detached target
`t = stop_gradient(r + gamma * bootstrap_mask * Z)`. It uses:

- Critic One: `0.5 * mean((Y - t) ** 2) + alpha * next_agreement`.
- Critic Two: `0.5 * mean((Z1 - t) ** 2) + alpha_prime * current_agreement`.

Critic Two's TD term therefore differentiates only through its current-input prediction Z1. During
the sequential Critic One phase, Z is freshly recomputed after the Critic Two optimizer step and is
then detached. Beta must remain disabled for this objective.

## Agreement ablations

The selectable agreement modes are:

- none: no agreement term.
- current: Critic Two adds alpha_prime * 0.5 * mean((Z1 - Y.detach()) ** 2).
- next: Critic One adds alpha * 0.5 * mean((Y1 - Z.detach()) ** 2).
- both: applies both terms in their respective critic phases.

Inactive coefficients must be zero. Agreement penalties can rotate critic updates away from shared error descent, so positive coefficients imply no convergence result.

## Beta ablations

Critic One can add beta_coefficient times one of:

- none: 0.
- linear_y: mean(Y), labeled experimental and unbounded below.
- l2_y: 0.5 * mean(Y ** 2).

l2_y_z is unavailable. With Z detached in Critic One's phase, its Z ** 2 component would not update Critic Two, so exposing that label would misstate the implemented regularization. A future symmetric Y/Z regularizer requires advisor approval and a separately named objective.

## Schedules

sequential applies Critic Two, Critic One, and actor updates in that order. Critic One recomputes Critic Two's bootstrap after the Critic Two step, and the actor uses the updated Critic Two.

simultaneous computes and stores all three parameter gradient sets from one pre-update snapshot. Optimizer steps begin only after all gradient sets are stored.

guarded uses sequential proposals and an update-batch shared error acceptance check. The guard can cover joint, critics, or actor. Every retry restores module parameters and optimizer states to the same snapshot, scales the relevant candidate learning rates by backtracking_factor ** retry, and checks error_after <= error_before + tolerance. A failed proposal is restored. Candidate rates do not accumulate across learner updates.

For a critics guard, the actor runs after the decision even if the critic proposal is rejected. For an actor guard, critic phases run before the actor snapshot and remain committed if the actor proposal is rejected. A joint rejection applies no phase.

When validation_batch is enabled, the trainer samples a second replay batch. Its shared error change is logged for monitoring and does not participate in acceptance.

## Diagnostics

Each learner update logs signed and absolute means for both delta and delta_prime, their squared errors, both raw agreement errors, component losses, gradient norms, actor objective, and statistics for Y, Y1, Z1, and Z.

Guard metrics include accepted and rejected indicators and totals, retry count, last executed candidate learning rate and scale, update-batch error before and after, committed error after restoration or acceptance, and validation-batch change when configured.

At a positive diagnostic_frequency, separate torch.autograd.grad calls measure:

- the inner product and cosine between Critic One's total objective and shared-error gradients;
- the same pair for Critic Two;
- the actor objective and shared-error gradient interaction with respect to actor parameters.

These calls do not write optimizer gradients or alter the training update.

## Deliberate deviations and limits

- The source expression omitted terminal handling. The implementation uses the explicit replay bootstrap mask and preserves separate terminated and truncated flags.
- All reductions use a batch mean and the shared error includes the specified factor of one half.
- Independent critic initialization, actor delay of one, and Adam remain fixed implementation constraints.
- Gaussian exploration is collection only. Evaluation is deterministic.
- Guard acceptance applies to one sampled batch. Validation-batch guard metrics are monitoring-only. Neither establishes replay-distribution descent or return improvement.
- No agreement, beta, guard, convergence, optimality, monotonic return, efficiency, or novelty claim follows from this implementation.

The implementation supplies controlled mechanisms and diagnostics for unit and smoke verification. Scientific comparison still requires planned multi-seed experiments and theory work.
