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

Beta modes are none, experimental linear_y, and l2_y. l2_y_z is unavailable because the current stop-gradient contract does not define symmetric Z regularization; any symmetric objective requires advisor approval and a separate name. Exact formulas are recorded in [notation](notation.md).

Every update records the residual, shared error, agreement errors, output statistics, component losses, and gradient norms. Guarded updates also record acceptance, rejection, retry count, last executed candidate scale and effective learning rate, batch error before and after, committed error, and optional validation change. At diagnostic_frequency, separate autograd calls measure each critic objective's alignment with its shared error gradient and the actor objective's interaction with the shared error gradient. Diagnostics never replace training gradients.

The next state actor objective remains the reconstructed default; current state input is an executable ablation. Critics remain independently initialized, the actor updates every learner step, and Adam is the implemented optimizer. See [controlled implementation](three_network_core.md). These mechanics establish no convergence or performance claim.
