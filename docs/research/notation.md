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

The original implementation remains `critic_objective=shared_error`. The opt-in
algorithm `three_network_shared_d` uses the same shared-D losses with both
positive agreement penalties, no beta and sequential critic updates. Its actor
update delay is configurable (default 2); D_prime is diagnostic only. This named
experimental schedule does not replace the historical implementation. Its
executable choices are preserved in the named configuration and manifest.

The separate `three_network_shared_d_conditioned` variant preserves those loss
signs and stop-gradient paths but uses asymmetric agreement coefficients:
`alpha=0.01` and `alpha_prime=10.0`. It also delays the first actor update until
5,000 critic updates have completed. The motivation, observed failure, local
conditioning calculation, and limits of this correction are recorded in
[shared-D instability correction](shared_d_instability_correction.md).

The opt-in `three_network_shared_d_two_timescale` variant restores equal
positive agreement coefficients and preserves the same four board vectors from
one transition minibatch. It changes only the optimizer clocks and Critic Two
step size: Q1 updates on every minibatch, Q2 updates less often and with a lower
learning rate, and the Actor is slower than Q2. Exact minibatch identities and
the declared schedule are preserved in the named configuration and manifest.

The follow-up `three_network_shared_d_board` reconstruction discards that
failed clock-decimation hypothesis. Both critics update from every sampled
transition minibatch in the source pseudocode order: Q2, recomputed Q2 outputs
for Q1, Q1, then Actor when enabled. It keeps D as both critics' shared residual
and keeps D_prime diagnostic-only. Actor start and independent-versus-identical
initial critic values are explicit screening controls; even with identical
initial values, the critics remain separately parameterized and optimized.

The named `three_network_shared_d_synchronous` diagnostic preserves the same D,
agreement terms, stop-gradient matrix, and absence of target networks. Its Q2
and Q1 gradients are both calculated from one pre-update critic snapshot; only
after both gradient sets exist are the two critic optimizer steps committed.
The slower Actor runs after both commits and may explicitly use current or next
replay states. The exact comparison remains reproducible from its manifest.

The separately named `three_network_shared_d_paired` variant follows the advisor's
2026-10-11 slow-Actor clarification. It preserves the shared-D losses above and
constructs all four vectors from the same ordered transition minibatch. Both
critic phases reuse one detached `actor(s_next)` action vector; Q2 values, not
the input rows, are refreshed after its update. Both critics update every
iteration, while the Actor defaults to every tenth critic-pair iteration
(`actor_update_delay=10`); delay 5 means two evenly spaced Actor updates per ten
pairs, not a two-step burst. Actor preparation is explicit and defaults to zero.
D_prime stays diagnostic-only.

The follow-up `three_network_shared_d_conditioned_paired` combines that exact
row-pairing contract with the earlier conditioned coefficients: `alpha=0.01`
and `alpha_prime=10.0`. It preserves both shared-D losses and prepares both
critics for 5,000 learner updates before its slower Actor clock starts. A
separate semantic marker prevents this configuration from being confused with
the failed equal-coefficient paired experiment. Its 20k one-seed early-signal screen
varies only the Actor delay among the proposed variants; the exact comparison
is preserved in its manifest.

The opt-in
`critic_objective=separate_td` reconstruction keeps the same Y, Y1, Z1, and Z definitions but
forms a detached target t = stop_gradient(r + gamma * m * Z). Its sequential critic losses are:

- L1 = 0.5 * mean((Y - t) ** 2) + 0.5 * alpha * mean((Y1 - stop_gradient(Z)) ** 2).
- L2 = 0.5 * mean((Z1 - t) ** 2) + 0.5 * alpha_prime * mean((Z1 - stop_gradient(Y)) ** 2).

For this objective, D remains r + gamma * m * Z - Y and the additional Critic Two residual is
D_prime = r + gamma * m * Z - Z1. The target is detached in both critic phases, so Critic Two's TD
gradient comes only from its current-input prediction Z1 and never through the bootstrap Z. Beta is
disabled and the update schedule is sequential. The actor objective and stop-gradient matrix are
unchanged.

The beta modes are:

- none: zero.
- linear_y: mean(Y), the original experimental proposal.
- l2_y: 0.5 * mean(Y ** 2).

The beta term belongs to Critic One's provisional objective. l2_y_z is unavailable: under the stop-gradient contract its Z ** 2 component would be constant during Critic One's phase and would not regularize Critic Two. A symmetric Y/Z regularizer would be a new, separately named objective and requires advisor approval. The linear term is unbounded below with respect to Y, and l2_y is a hypothesis rather than a proved fix.

Agreement mode may be none, current, next, or both. alpha_prime weights current agreement in Critic Two and alpha weights next agreement in Critic One. See the [stop gradient matrix](architecture.md).
