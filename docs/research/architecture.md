# Architecture and gradient contract

The proposed main variant has exactly one deterministic actor and two independently optimized critics. It has no target actor or target critic. A target-network variant can be introduced only as a named ablation. DDPG and TD3 retain their separate baseline target settings.

| Update | Parameters receiving gradients | Fixed parameters | Required path |
| --- | --- | --- | --- |
| Critic Two | `φ′` | `θ`, `φ` | `Z` and `Z1` through Critic Two; detach `Y` |
| Critic One | `φ` | `θ`, `φ′` | `Y` and `Y1` through Critic One; detach `Z` |
| Actor | `θ` | `φ`, `φ′` | Gradient through Critic Two's action input into actor |

Freezing Critic Two's parameters during actor ascent must **not** detach its output from the actor action. Each optimizer clears and steps only its own parameters. The simultaneous schedule computes all gradients from one parameter snapshot before any optimizer step. A sequential schedule recomputes outputs after earlier steps when its definition requires it. Guarded steps need an explicit acceptance batch, tolerance, backtracking policy, and optional independent validation batch.

The common algorithm interface covers action selection, observation of either transition batches or rollouts, updates, evaluation mode, state serialization, and checkpoint restoration. Capability metadata distinguishes on-policy algorithms from replay-based algorithms. This interface does not dictate a shared trainer or loss. Training, evaluation, wrappers, algorithms, tracking, and plotting should remain separate components.
