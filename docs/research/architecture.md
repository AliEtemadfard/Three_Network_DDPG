# Architecture and gradient contract

The proposed main variant has exactly one deterministic actor and two independently optimized critics. It has no target actor or target critic. A target-network variant can be introduced only as a named ablation. DDPG and TD3 retain their separate baseline target settings.

| Update | Parameters receiving gradients | Fixed parameters | Required path |
| --- | --- | --- | --- |
| Critic Two | `φ′` | `θ`, `φ` | `Z` and `Z1` through Critic Two; detach `Y` |
| Critic One | `φ` | `θ`, `φ′` | `Y` and `Y1` through Critic One; detach `Z` |
| Actor | `θ` | `φ`, `φ′` | Gradient through Critic Two's action input into actor |

Freezing Critic Two's parameters during actor ascent must **not** detach its output from the actor action. Each optimizer clears and steps only its own parameters. The simultaneous schedule computes all gradients from one parameter snapshot before any optimizer step. A sequential schedule recomputes outputs after earlier steps when its definition requires it. Guarded steps need an explicit acceptance batch, tolerance, backtracking policy, and optional independent validation batch.

The common algorithm interface covers action selection, observation of either transition batches or rollouts, updates, evaluation mode, state serialization, and checkpoint restoration. Capability metadata distinguishes on-policy algorithms from replay-based algorithms. This interface does not dictate a shared trainer or loss. Training, evaluation, wrappers, algorithms, tracking, and plotting should remain separate components.

## Executable core boundary

`Transition` stores one Gymnasium step with independent `terminated` and `truncated` flags. `ReplayBuffer` copies transitions into a fixed-capacity ring and samples float32 tensors for observations, actions, rewards, and the explicit bootstrap mask; end flags stay boolean. `bootstrap_on_truncation` controls whether a time-limit transition retains its next-state value. Terminal transitions never bootstrap. The learner still owns its Bellman target and loss.

`ActionSelection` carries an environment-scale continuous action and an `exploration_applied` flag. `scale_action` and `unscale_action` map between normalized and environment bounds; `clip_action` is a separate explicit operation. The trainer's `clip_actions` setting controls clipping at the environment boundary. The evaluator requests deterministic actions and rejects selections marked as exploratory.

`OffPolicyAlgorithm` accepts individual transitions and replay batches. `OnPolicyAlgorithm` accepts rollout batches through separate methods. `OffPolicyTrainer` checks capabilities and refuses planned or on-policy algorithms; it contains environment interaction and update scheduling but no loss logic. The registry exposes the reviewed DDPG baseline as `verified`; all other algorithms remain planned. Unit tests exercise shared contracts with a no-learning test double and DDPG with focused gradient and target tests. The tiny deterministic environment and Pendulum-v1 factory support short interaction checks. The evaluator is separate and must be given its own environment instance.

`prepare_run_directory` writes the resolved configuration and provenance before a future run. The DDPG entry point adds JSONL tracking, a summary, and a checkpoint. Future study tooling must add final run status and multi-seed aggregation. None of these core additions alters the proposed three-network gradient contract above.
