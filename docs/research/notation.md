# Notation and provisional objectives

For a replay minibatch (s, a, r, s_next, terminated, truncated), the deterministic actor is mu_theta, Critic One is Q_phi, and Critic Two is Q_phi_prime. Let a_next = mu_theta(s_next), detached during critic updates. The four vectors have batch length B:

| Vector | Definition | Role |
| --- | --- | --- |
| Y | Q_phi(s, a) | Current side of shared residual |
| Y1 | Q_phi(s_next, a_next) | Next input agreement |
| Z1 | Q_phi_prime(s, a) | Current input agreement |
| Z | Q_phi_prime(s_next, a_next) | Bootstrap and actor objective |

Use a documented mask m in delta = r + gamma * m * Z - Y and diagnostic e = 0.5 * mean(delta ** 2). A terminal transition has m = 0. For truncations, bootstrap_on_truncation explicitly selects whether to bootstrap; the final observation must be available when bootstrapping. Never infer truncation from termination. The source board expression omitted this mask.

Current agreement is c_current = 0.5 * mean((Z1 - Y) ** 2). Next agreement is c_next = 0.5 * mean((Y1 - Z) ** 2). For Critic Two, Y is detached and the provisional loss is e + alpha_prime * c_current. For Critic One, Z is detached and the provisional loss is e + alpha * c_next + beta_coefficient * beta_regularizer. The actor maximizes mean(Q_phi_prime(s_next, mu_theta(s_next))) in the board interpretation; current state input is an ablation. All implemented reductions use a batch mean.

The beta modes are:

- none: zero.
- linear_y: mean(Y), the original experimental proposal.
- l2_y: 0.5 * mean(Y ** 2).

The beta term belongs to Critic One's provisional objective. l2_y_z is unavailable: under the stop-gradient contract its Z ** 2 component would be constant during Critic One's phase and would not regularize Critic Two. A symmetric Y/Z regularizer would be a new, separately named objective and requires advisor approval. The linear term is unbounded below with respect to Y, and l2_y is a hypothesis rather than a proved fix.

Agreement mode may be none, current, next, or both. alpha_prime weights current agreement in Critic Two and alpha weights next agreement in Critic One. See the [stop gradient matrix](architecture.md).
