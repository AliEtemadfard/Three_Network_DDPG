# Architecture and gradient contract

The proposed main variant has exactly one deterministic actor and two independently optimized critics. It has no target actor or target critic. A target network variant can be introduced only as a named ablation. DDPG and TD3 retain their separate baseline target settings.

| Update | Parameters receiving gradients | Fixed parameters | Required path |
| --- | --- | --- | --- |
| Critic Two | phi_prime | theta, phi | Z and Z1 through Critic Two; detach Y |
| Critic One | phi | theta, phi_prime | Y and Y1 through Critic One; detach Z |
| Actor | theta | phi, phi_prime | Gradient through Critic Two's action input into actor |

Freezing Critic Two during actor ascent must preserve the path from its action input to the actor. Each optimizer clears and steps only its own parameters.

## Shared contracts

Transition stores one Gymnasium step with separate terminated and truncated flags. ReplayBuffer samples float32 observations, actions, rewards, and an explicit bootstrap mask while end flags remain boolean. bootstrap_on_truncation controls whether time limit transitions retain their next state value. Terminal transitions never bootstrap.

ActionSelection carries an environment scale continuous action and an exploration_applied flag. The trainer controls explicit clipping at the environment boundary. The evaluator requests deterministic actions and rejects exploratory selections.

OffPolicyAlgorithm accepts transitions and replay batches. OnPolicyAlgorithm accepts rollout batches through separate methods. OffPolicyTrainer checks capabilities and refuses planned or on-policy algorithms; it owns environment interaction and update timing but no loss logic. An off-policy implementation can request an independent validation replay batch through the update contract. A3C remains outside this loop.

## DDPG without a target actor

`ddpg_no_target_actor` is a controlled DDPG baseline ablation and is not part of the proposed three-network method. It retains the online actor, online critic, and target critic, and has no target actor module. Its critic bootstrap is `r + gamma * bootstrap_mask * Q_target(s_next, actor_online(s_next))`. The next action and target value are detached during critic regression, so this phase cannot update the actor. The actor still maximizes the online critic on current states, and Polyak averaging updates only the target critic.

The ablation uses the shared off-policy trainer, replay buffer, evaluator, action scaling, and DDPG network components. Its registry name, resolved config, run metadata, and checkpoint identity are `ddpg_no_target_actor`. This executable contract and smoke coverage establish implementation behavior only; stability and return require controlled multi-seed comparisons.

## Three-network update schedules

sequential applies Critic Two, recomputes the needed outputs, applies Critic One, then applies the actor. The later phases intentionally see earlier parameter changes.

The separately registered `three_network_shared_d` variant reuses the shared-error
critic phases with both agreement penalties and no beta. It runs the actor only
every configured `actor_update_delay` critic iterations (default 2), after Q1.
On intervening iterations only Q2 and Q1 step; all gradient paths in the matrix
above are unchanged. D_prime is logged but never used as its training loss.
Historical `three_network_ddpg` variants still update the actor every iteration.

`three_network_shared_d_conditioned` is a distinct follow-up to the failed long
shared-D run. It preserves the same critic equations and gradient matrix, uses
stronger current than next agreement, and performs an explicit critic-only
preparation phase before the delayed Actor begins. Its checkpointed update clock
determines both preparation and delay, so resume cannot repeat the warm-up.

`three_network_shared_d_two_timescale` retains the board-defined shared-D losses
but uses separate clocks. Q1 updates on every learner minibatch; Q2 updates on a
configurable slower cadence and with a lower configured learning rate; Actor
updates are counted in completed Q2 updates after an explicit preparation
period. When Q2 and Actor are due, sequential order remains Q2, recomputed Q1,
Actor. Every due phase uses the same transition minibatch, with current
`(s, a)` and next `(s_next, actor(s_next))` views kept distinct.

`three_network_shared_d_board` is the follow-up after the slower and less
frequent Q2 schedule failed its one-seed screen. It returns to the source
pseudocode: Q2 and Q1 both update on every learner minibatch, Q2 is recomputed
before Q1, and the Actor is last. It permits only two named screening controls:
an Actor-only learning start and identical initial critic values. Identically
initialized critics are distinct modules with distinct optimizers; no
parameters are shared. Target networks, beta, D_prime training, and Critic Two
clock decimation are excluded.

`three_network_shared_d_synchronous` is a named stability diagnostic, not a
replacement for the sequential board reconstruction. It calculates Q2 and Q1
gradient sets from the same pre-update parameter snapshot while retaining the
shared-D and agreement gradient paths in the matrix above. Both critic updates
are committed before an independently delayed Actor update. This makes the
critic timing different from the historical `simultaneous` schedule, which
also calculated the Actor gradient from the old snapshot. Current-versus-next
Actor state input is an explicit screening factor. Target networks remain
absent and D_prime remains monitoring-only.

`three_network_shared_d_paired` retains the sequential Q2 -> fresh Q2 values ->
Q1 -> due Actor order and the stop-gradient matrix unchanged. It holds a single
ordered TransitionBatch and one detached next-action vector for both critic
phases: Y/Z1 use matched `(s_i, a_i)` and Y1/Z use matched
`(s_next_i, actor(s_next_i))`, with r_i and mask_i from that transition. The
Actor uses attached actions on those same next-state rows. No phase resamples
or independently permutes a batch column. Both critics update on every learner
iteration; Actor delay 10 (or 5) implements one (or two evenly spaced) Actor
updates per ten critic-pair iterations. The default extra Actor warm-up is zero,
separate from collection warm-up. A `board_aligned_v1` checkpoint semantic marker
and strict schedule compatibility prevent silent reuse of older checkpoints.
These are explicit input/schedule contracts, not evidence that older replay rows
were misaligned or that the new version converges.

`three_network_shared_d_conditioned_paired` preserves the same paired inputs,
update order, and stop-gradient paths, but restores the earlier asymmetric
agreement anchor (`alpha=0.01`, `alpha_prime=10.0`) and 5,000-update critic
preparation period. Its named semantic contract fixes those values while
allowing Actor delays 2, 5, and 10 to be compared without changing any other
proposed-method setting. This is a recovery hypothesis motivated by observed
Q drift; it is not a convergence guarantee.

simultaneous computes and stores all three parameter gradient sets from one pre-update parameter snapshot. No optimizer steps occur before all required gradients are stored.

guarded builds sequential proposals. It snapshots all three modules and all three optimizer states before the guarded phase, evaluates shared squared error on the update batch, and either accepts, retries from the same snapshot with a smaller effective learning rate, or restores the snapshot. Backtracking rates apply only to the candidate and reset to configured rates for the next learner update.

Guard scope is explicit:

| Scope | Guarded proposal | Outside the guard |
| --- | --- | --- |
| joint | Critic Two, Critic One, actor | none |
| critics | Critic Two and Critic One | actor runs after the guard decision |
| actor | actor | critics run before the guarded actor phase |

For critics and actor scope, rejection restores only the guarded proposal because the other phase intentionally remains committed. The metrics distinguish the guard decision from whether any phase was applied. An optional independent validation batch is monitoring only and never controls acceptance.

## Controlled objectives and diagnostics

Current agreement compares Z1 with detached Y during Critic Two's update. Next agreement compares Y1 with detached Z during Critic One's update. Modes none, current, next, and both select these terms without changing the stop gradient matrix.

The default `shared_error` critic objective preserves the original implementation and all earlier
results. The opt-in `separate_td` reconstruction is sequential-only: Critic Two regresses its current
prediction Z1 against the detached target `r + gamma * mask * Z`, then Critic One recomputes Z after
that update and regresses Y against the same-form detached target. Agreement terms retain their
existing current and next inputs. No critic gradient flows through Z as a bootstrap value.

Beta modes are none, experimental linear_y, and l2_y. l2_y_z is unavailable because the current stop-gradient contract does not define symmetric Z regularization; any symmetric objective requires advisor approval and a separate name. Exact formulas are recorded in [notation](notation.md).

Every update records the shared residual, Critic Two's own TD residual, both squared errors, agreement errors, output statistics, component losses, and gradient norms. Guarded updates also record acceptance, rejection, retry count, last executed candidate scale and effective learning rate, batch error before and after, committed error, and optional validation change. At diagnostic_frequency, separate autograd calls measure each critic objective's alignment with its shared error gradient and the actor objective's interaction with the shared error gradient. Diagnostics never replace training gradients.

At that same diagnostic interval, Actor outputs on current and next replay
states are normalized by the environment action half-range and logged as mean
and maximum absolute utilization. These monitoring-only values have no hidden
saturation threshold and never affect an update.

The next state actor objective remains the reconstructed default; current state input is an executable ablation. Critics remain independently initialized, the actor updates every learner step, and Adam is the implemented optimizer. See [controlled implementation](three_network_core.md). These mechanics establish no convergence or performance claim.
