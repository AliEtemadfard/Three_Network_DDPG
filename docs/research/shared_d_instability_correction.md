# Shared-D long-run diagnosis and conditioned correction

## Observed failure

The completed `pendulum-shared-d-long-01` study contains one finite, completed
100,000-step run per variant. DDPG's final five-seed evaluation mean was
`-190.646`; shared-D's was `-1374.011`. DDPG's last-ten checkpoint mean was
`-191.162`, while shared-D's was `-1374.011`. Shared-D never improved beyond its
initial `-1354.654` checkpoint and became exactly constant at the failed-control
return after the early evaluations.

The earliest mechanism difference precedes that return plateau. At environment
step 1,100, shared error had fallen from `19.239` to `1.994`, but mean outputs
were already `Y=-2.771`, `Z_current=2.241`, and `Z_next=2.724`; DDPG's Q mean was
`-7.214`. Thus, reducing D initially moved Q2 in the positive direction while
DDPG propagated the negative Pendulum reward into its current prediction.

The shared-D outputs then moved together to the wrong scale. At step 5,000,
mean absolute Q was `2,893.5`; at 20,000 it was `152,740`; at 100,000 it was
`28,045,000`. Final shared error was `2.074e9`, Q2's diagnostic own residual was
`6.796e10`, current agreement was `5.067e10`, and next agreement was `6.224e10`.
Critic gradient norms reached `3.53e10` and `2.85e10` before the configured norm
clip. Actor gradient reached zero by step 20,000. On a fixed grid of 5,973 valid
Pendulum observations, the final shared-D actor emitted exactly `-2.0` for every
observation; DDPG retained both action signs. This establishes actor saturation
for the inspected grid, not for every mathematically possible observation.

Both runs used the intended resolved settings: raw Pendulum-v1, gamma 0.99,
reward scale 1, bootstrap across truncations, seed 11, five fixed evaluation
seeds, 128x128 ReLU networks, Adam, actor rate 1e-4, critic rates 3e-4, gradient
clip 10, and 100,000 steps. There was no configuration mismatch, non-finite
metric, failed process, target-network leak, D-prime training term, or wrong
terminal mask explaining the difference.

## Formula diagnosis

The implemented signs match the positive squared objectives:

\[
D=r+\gamma mZ_+-Y,\quad
L_1=\tfrac12 E[D^2]+\tfrac{\alpha}{2}E[(Y_+-Z_+)^2],
\]

\[
L_2=\tfrac12 E[D^2]+\tfrac{\alpha'}{2}E[(Z_{current}-Y)^2].
\]

D-prime remains diagnostic only. The failure is therefore not evidence for a
sign typo in code. It exposes a conditioning problem in this formulation. If a
constant c is added to all four critic outputs, both agreement errors are
unchanged, while

\[
D\;\mapsto\;D-(1-\gamma m)c.
\]

For nonterminal Pendulum samples, gamma=0.99, so the squared shared residual has
only `(1-gamma)^2 = 1e-4` curvature in this common translation direction. The
agreement penalties cannot anchor that direction because they see only
differences. This is consistent with prior work identifying mean-squared Bellman
error as an ill-conditioned objective; it does not transfer that paper's
guarantees to this actor-critic system.

For the deliberately simplified one-state, nonterminal, fixed-policy scalar
system, the output-gradient dynamics have Jacobian

\[
A=\begin{bmatrix}
-(1+\alpha)&\gamma+\alpha\\
\gamma+\alpha'&-(\gamma^2+\alpha')
\end{bmatrix},\qquad
\det(A)=(1-\gamma)(\alpha'-\gamma\alpha).
\]

This is a diagnostic local model, not a convergence proof. With the old
alpha=alpha-prime=0.1, its eigenvalues are `-2.180095` and `-4.587e-6`: one mode
is nearly neutral. This explains why equal-looking coefficients are not a
neutral default when gamma is near one. The observed common Q drift is the same
mode predicted by this calculation.

## Conditioned experimental correction

Historical `three_network_shared_d` code and artifacts remain unchanged. The new
registry name is `three_network_shared_d_conditioned`. It retains D in both
critic losses, positive agreement penalties, Q2 -> fresh Q1 -> Actor ordering,
three independent networks and optimizers, no target networks, no beta, and the
existing stop-gradient matrix. It changes only the previously unsupported
conditioning and policy-start assumptions:

- `alpha=0.01` for next agreement;
- `alpha_prime=10.0` for current agreement;
- 5,000 critic-only learner updates before any Actor update;
- Actor then updates every two critic iterations.

The scalar diagnostic eigenvalues become `-11.981762` and `-0.008338`, improving
the slow-mode magnitude by about 1,800 times. Strong current coupling is intended
to keep Q2 tied to Q1's reward-driven current prediction; weak next coupling
avoids making Q1 immediately chase an unanchored positive bootstrap. The
critic-only phase prevents the actor from exploiting that transient bootstrap.
These are explicit development hypotheses. They do not prove that the nonlinear,
off-policy algorithm is stable or that it will outperform DDPG.

The isolated manual manifest is
`experiments/manifests/pendulum_shared_d_conditioned_01.yaml`. It schedules DDPG
and the conditioned variant with seed 11 for 100,000 steps each and writes only
under `runs/pendulum-shared-d-conditioned-01`. The failed study is never reused
or overwritten.

Primary background: [Sharifnassab and Sutton, 2023](https://proceedings.mlr.press/v202/sharifnassab23a.html).

## Manual commands

These commands use the repository launcher, so an active `.venv` cannot shadow
the canonical Conda interpreter. Validation and planning do not train. The `run`
line is intentionally manual and was not executed while preparing this change.

```powershell
Set-Location 'F:\lessons\arshad\three-network-ddpg'
.\Experiment.cmd validate experiments\manifests\pendulum_shared_d_conditioned_01.yaml
.\Experiment.cmd plan experiments\manifests\pendulum_shared_d_conditioned_01.yaml
.\Experiment.cmd run experiments\manifests\pendulum_shared_d_conditioned_01.yaml --workers 1
.\Experiment.cmd status experiments\manifests\pendulum_shared_d_conditioned_01.yaml
.\Experiment.cmd aggregate experiments\manifests\pendulum_shared_d_conditioned_01.yaml
.\Experiment.cmd plot experiments\manifests\pendulum_shared_d_conditioned_01.yaml --metric evaluation_return_mean
.\Experiment.cmd plot experiments\manifests\pendulum_shared_d_conditioned_01.yaml --metric shared_squared_error
.\Experiment.cmd plot experiments\manifests\pendulum_shared_d_conditioned_01.yaml --metric q_abs_mean
.\Experiment.cmd plot experiments\manifests\pendulum_shared_d_conditioned_01.yaml --metric current_agreement_error
.\Experiment.cmd plot experiments\manifests\pendulum_shared_d_conditioned_01.yaml --metric next_agreement_error
```

Use `resume` only after inspecting an interrupted or failed attempt; it starts a
new attempt rather than continuing the replay buffer of the old attempt.
