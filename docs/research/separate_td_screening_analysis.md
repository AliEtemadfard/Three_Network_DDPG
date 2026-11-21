# Separate-TD Pendulum screening analysis

Analysis date: 2026-10-06. This report examines existing artifacts only. No training,
environment stepping, or resume was performed for this analysis.

## Finding

The opt-in `separate_td` implementation matches the requested equations with **positive**
agreement penalties (the user confirmed that the hyphens were list formatting). No
equation-level implementation error was found, and no algorithm change is justified by
this audit. Compared with the original shared-error objective, this one-seed screening
shows a substantial reduction in value drift, residuals, disagreement, and critic gradients
throughout much of the trajectory, not merely a better final return. However, `separate_td`
still has an early critic spike, oscillating control performance, and much worse final
return than DDPG. It has better observed numerical behavior than the original in this
screening; it has not established stability, convergence, or a reproducible advantage.

The strongest working explanation is rapid bootstrap feedback through online Q2, with
critic step sizes inherited from target-network DDPG. Actor/critic interaction remains a
plausible contributor. Neither explanation is isolated by the available experiment.

## Artifact provenance and executed matrix

The latest completed screening is
`runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01`.
All three runs used commit `e33269c4deb12144f8af63fac07559d601c29e93`, attempt `0001`,
training seed **11**, and evaluation seeds **1001, 1002, 1003, 1004, 1005**.
Each completed 5,000 training environment steps, 4,744 learner updates, 25 training
episodes, and 10 evaluations of five 200-step episodes. Status and completion markers
agree; all logged numeric values are finite. There are zero failed, non-finite, divergent,
interrupted, or incomplete runs in this screening. These are execution classifications,
not scientific stability criteria.

| Variant | Run ID | Config hash |
| --- | --- | --- |
| DDPG | `ddpg-seed-11-b31941eda6c5` | `881710da634298d744daae2ea7f7a957120acf52855fd4f05feca2a6c0113efe` |
| Original shared error | `three-network-shared-error-seed-11-7831a1794f66` | `4afd90a9547cb2d495c3376fb397295390392605bfd9fa51133ed9f520125571` |
| Separate TD | `three-network-separate-td-seed-11-a4817cf17ad5` | `cd806ba6f88a6db1817e0d1284f54c888fc78dcc7ab8f208b35c426f69994be0` |

Evidence: the [screening manifest](../../experiments/manifests/pendulum_separate_td_screening_01.yaml),
each run's `resolved_config.yaml` and `run_spec.json`, and each attempt's
`metadata.json`, `summary.json`, `metrics.jsonl`, `COMPLETED.json`, and `checkpoint.pt`.
Resolved YAML was checked against the configuration saved in metadata. Checkpoint
network shapes, optimizer groups, and target-network presence were inspected directly.
The metadata records Python 3.11.16, Torch 2.14.0, Gymnasium 1.3.0, NumPy 2.4.6,
CPU execution on an Intel Family 6 Model 140 processor, and four logical CPUs.

Raw `metrics.jsonl` SHA-256 identifiers, in the same variant order:

```text
cfc1a89f3a9c36596cca93ce3d24bafa73620a94b5ca574783598b5e854caf40
155073cf7680112e7d22e8ac8c0b79e9a5b46305c514716c6a85282718d3a5d6
5284f628e2e4526f66160ce59e09d89f6f6fb2503781d06ce404f6a787bc2cce
```

The shared-error comparator here has **both agreement penalties at 0.1**. It is not the
agreement-disabled Stage B reference. Earlier pilot results are documented separately in
[the pilot failure analysis](pendulum_pilot_01_failure_analysis.md).

## Equation and gradient audit

Let `sg` denote stop-gradient, and let the expectation below be the minibatch mean:

\[
a'=\operatorname{sg}(\mu_\theta(s')),\quad
Y=Q_1(s,a),\quad Y'=Q_1(s',a'),\quad
Z_c=Q_2(s,a),\quad Z_n=Q_2(s',a'),
\]

\[
t=\operatorname{sg}(r+\gamma m Z_n),\qquad
D=r+\gamma mZ_n-Y,\qquad D'=r+\gamma mZ_n-Z_c.
\]

\[
L_1=\tfrac12\mathbb E[(Y-t)^2]
       +\tfrac{\alpha}{2}\mathbb E[(Y'-\operatorname{sg}(Z_n))^2],
\]

\[
L_2=\tfrac12\mathbb E[(Z_c-t)^2]
       +\tfrac{\alpha'}{2}\mathbb E[(Z_c-\operatorname{sg}(Y))^2],
\qquad L_A=-\mathbb E[Q_2(s',\mu_\theta(s'))].
\]

The implementation uses `critic_objective=separate_td`, `agreement_mode=both`,
`alpha=alpha_prime=0.1`, sequential updates, next-state actor input, and beta mode
`none` with coefficient zero. It has three independently parameterized networks and
three distinct Adam optimizers, without target networks.

| Requirement | Actual code and gradient consequence |
| --- | --- |
| Actor fixed in critic phases | `_next_actions_detached` computes actor actions under `torch.no_grad()`. |
| Q2 current prediction learns TD | `_sequential_critic_two` uses `0.5 * mean((z_current - target)^2)` as its base loss. `target` is detached after forming the bootstrap expression. |
| Other critic detached in Q2 phase | Q1's current output is computed under `no_grad`; Q2 agreement differentiates only `z_current`. |
| Fresh Q2 for Q1 | `_sequential_update` steps Q2 first. `_sequential_critic_one` then calls Q2 again under `no_grad`; it does not reuse the pre-Q2 bootstrap tensor. |
| Q1 next-input agreement active | Q1's current and next outputs are fresh differentiable forwards; Q2's next value is fixed. |
| Only phase parameters updated | Each phase clears gradients, calls `autograd.grad` for only its own parameter tuple, assigns those gradients, and steps only its own optimizer. |
| Actor gradient through Q2 action | `_actor_loss` freezes both critics' parameters but leaves the Q2 output connected through its action argument to the actor. The loss uses next states. |
| Order and sampling | Q2 then Q1 then Actor, once per learner update, on the same replay minibatch. |

Relevant sources: [three-network implementation](../../src/three_network_ddpg/algorithms/three_network_ddpg/implementation.py),
[trainer](../../src/three_network_ddpg/training.py), and
[existing independent-reference tests](../../tests/unit/test_three_network_separate_td.py).
The six tests in that focused file passed during this audit. They cover Q2 current TD
gradients with zero agreement, rejection of the undetached bootstrap gradient, Q1
recomputation and next-agreement gradients, fixed non-updated modules, the actor action
gradient path, metrics, and manifest/configuration constraints. No new regression test
was added because no objective violation was found.

Writing Jacobians with respect to the phase's own parameters, the implemented Q2
gradient is

\[
\nabla_{\phi_2}L_2
=-\mathbb E[D'J_{Z_c}]
 +\alpha'\mathbb E[(Z_c-\operatorname{sg}(Y))J_{Z_c}].
\]

There is **no** `gamma * mask * J_Zn` TD gradient. A differentiated full-residual
objective would instead contain
`E[D_prime * (gamma * mask * J_Zn - J_Zc)]`, plus agreement. That is a different
algorithm, not an omitted derivative in the specified TD-target implementation.
The original shared-error Q2 base gradient is `E[D * gamma * mask * J_Zn]`;
its update can change the bootstrap value directly to reduce D rather than fit the
current Q2 prediction to its own Bellman target.

## Fixed settings and important baseline differences

| Setting | Executed value |
| --- | --- |
| Environment and wrappers | `Pendulum-v1`, no extra wrappers; native 200-step time limit |
| Observation/action | Three raw observations; scalar torque; actor tanh rescaled to [-2, 2]; collected actions clipped |
| Normalization/reward | Observation normalization false; reward scale 1.0 |
| Architecture | Actor 3 → 128 ReLU → 128 ReLU → 1 tanh; critics concatenate observation/action, 4 → 128 ReLU → 128 ReLU → 1 linear |
| Network parameter counts | Actor 17,153; each critic 17,281. Three-network total 51,715; DDPG online total 34,434, with another 34,434 in target copies |
| Initialization | PyTorch Linear default initialization under seed 11; actor and critics constructed successively. Critics share architecture, not parameters or identical initial weights |
| Optimizers | Separate Adam, betas (0.9, 0.999), epsilon 1e-8, weight decay zero |
| Learning rates | Actor 3e-4; each online critic 1e-3 |
| Discount/mask | Gamma 0.99; zero bootstrap for termination; bootstrap retained for truncation using the actual next observation |
| Replay | Capacity 10,000; batch 64 sampled without replacement within each batch; warm-up 256 uniform random actions |
| Collection/update | One replay update per environment step after warm-up; actor updated every learner update |
| Exploration | Independent Gaussian noise, standard deviation 0.1 in environment action units; no noise clip before final action clipping |
| Evaluation | Every 500 training steps; five fixed seeds 1001–1005; separate environment; deterministic actor; no exploration |
| Diagnostics | Scalar mechanism/gradient metrics each update; alignment diagnostics every 100 learner updates; no gradient clipping, guard, or validation batch |
| Execution | CPU, one worker, sequential runs |

DDPG uses a detached target actor and target critic with Polyak coefficient 0.005,
and a current-state actor objective. Its critic minimizes `mean(TD_error^2)`, while
the three-network base losses have a factor 0.5. Raw gradient norms therefore differ
in loss scaling as well as model and data distribution. Adam partly normalizes constant
gradient scaling, but this does not make the trajectories equivalent. The critics also
see different replay data once the policies differ, despite paired random seeds.

The summary's metric named `parameter_count` is not a network parameter count:
`experiment._tensor_parameter_count` counts tensors recursively in the complete
checkpoint, including Adam moments, counters, and buffers. For example, the
three-network summary reports 155,165. The architecture counts above use network
weight and bias tensors only. This logging issue does not violate the requested
equations and was left unchanged in this report-only audit.

## Returns, learning curves, and runtime

Evaluation returns are means over the same five evaluation initializations; higher is better.

| Training steps | DDPG | Original shared error | Separate TD |
| ---: | ---: | ---: | ---: |
| 500 | -1753.583 | -1374.817 | -1769.815 |
| 1000 | -1479.695 | -1374.011 | -1377.464 |
| 1500 | -1378.016 | -1374.011 | -1179.150 |
| 2000 | -1265.828 | -1374.011 | -1382.418 |
| 2500 | -1094.083 | -1374.011 | -1494.329 |
| 3000 | -1212.142 | -1374.011 | -1457.239 |
| 3500 | -856.196 | -1374.011 | -1280.287 |
| 4000 | -494.923 | -1374.011 | -1073.966 |
| 4500 | -202.774 | -1374.011 | -1069.609 |
| 5000 | -196.603 | -1374.011 | -1017.240 |

| Summary | DDPG | Original shared error | Separate TD |
| --- | ---: | ---: | ---: |
| Final evaluation return | -196.603 | -1374.011 | -1017.240 |
| Last-five-evaluation mean | -592.528 | -1374.011 | -1179.668 |
| Normalized AUC, observed 500–5000 range | -995.417 | -1374.055 | -1300.888 |
| Wall time, seconds | 150.623 | 723.809 | 726.069 |
| Training steps/second, including evaluation overhead | 33.195 | 6.908 | 6.886 |

Normalized AUC is trapezoidal area divided by the observed step span, not reward
normalization or a percentage. There is no step-zero evaluation or extrapolation.
With one training seed, the aggregate standard deviation of zero is bookkeeping;
no between-training-seed variance estimate or confidence interval is available.
Evaluation episodes are not independent training replications.

At the final checkpoint, individual evaluation-seed returns were:

| Evaluation seed | DDPG | Original shared error | Separate TD |
| ---: | ---: | ---: | ---: |
| 1001 | -133.137 | -803.893 | -896.039 |
| 1002 | -125.864 | -1668.681 | -901.780 |
| 1003 | -115.911 | -1595.039 | -941.206 |
| 1004 | -346.166 | -1484.714 | -1253.443 |
| 1005 | -261.937 | -1317.726 | -1093.732 |

Separate TD is better than the original on four of five final evaluation seeds, but
worse on seed 1001. Its early advantage at 1,500 steps reverses at 2,000–3,000 steps
before later recovery. DDPG overtakes it by 2,000 steps and finishes far ahead.
The original reaches an almost constant poor return while its internal critic scale
continues growing. Constant return is not convergence evidence.

## Residuals, agreement, value scale, and gradients

Cells below give **final / maximum observed** values over 4,744 learner updates.
`shared_squared_error` and `q2_td_squared_error` both mean half the mean squared
residual, despite their names. Agreement errors have the same factor 0.5 and are
unweighted; the training penalty multiplies them by 0.1.

| Metric | Original shared error | Separate TD |
| --- | ---: | ---: |
| D signed mean | -27.094 / largest absolute signed mean -219.290 | -0.215 / largest absolute signed mean 28.496 |
| Mean absolute D | 85.304 / 456.950 | 11.157 / 51.069 |
| Half mean D² | 13,109.763 / 325,903.031 | 88.684 / 2,126.250 |
| D′ signed mean | -56.826 / largest absolute signed mean 517.319 | -1.226 / largest absolute signed mean -24.596 |
| Mean absolute D′ | 622.106 / 1,497.718 | 8.543 / 46.239 |
| Half mean D′² | 322,233.063 / 1,802,938.750 | 56.838 / 1,838.125 |
| Current agreement error | 246,620.766 / 1,131,655.000 | 19.973 / 153.605 |
| Next agreement error | 244,951.578 / 873,541.938 | 11.537 / 199.063 |
| Mean absolute Q, all four outputs | 14,085.712 / 23,315.547 | 372.072 / 706.158 |
| Maximum absolute Q, all four outputs | 25,565.396 / 34,001.914 | 502.898 / 1,373.998 |
| Actor gradient norm | 0 / 2.454 | 10.767 / 125.054 |
| Critic One gradient norm | 63,899.891 / 764,374.688 | 859.719 / 13,684.043 |
| Critic Two gradient norm | 79,267.883 / 803,583.938 | 519.521 / 10,311.163 |

DDPG final/peak maximum absolute Q was **151.673 / 152.787**, actor gradient norm
**2.555 / 8.347**, and critic gradient norm **82.801 / 181.558**. DDPG's own TD MSE
was **1.742 / 53.254** (half-MSE would be **0.871 / 26.627**). D and D′ involving
two critics, agreement errors, and mean absolute Q were not logged for DDPG and
cannot be directly reconstructed from its saved scalar metrics. DDPG's Q maximum
covers its current online critic prediction; the three-network maximum spans four
outputs, so their maxima are not identical estimands. The Q1 current-input comparison
below avoids that particular scope difference.

The original's final mean Q1 and Q2-next predictions are **+14,048.327** and
**+14,169.595**, whereas Separate TD's are **-373.058** and **-371.414**.
Pendulum rewards are nonpositive under the unchanged reward scale. Large positive
values in the original are inconsistent with the discounted return of a nonpositive
reward process; this is stronger evidence of a value-estimation pathology than its
finite completion status. Negative hundreds in Separate TD are not automatically
wrong: poor policies and gamma 0.99 can produce that scale. Calibration would require
matched state/action return targets, which were not stored here.

Separate TD's last 500 updates average half-D² **136.238**, half-D′² **70.708**,
current/next agreement **28.035 / 27.771**, mean absolute Q **351.856**, and
Q1/Q2 gradient norms **892.074 / 549.771**. These show persistent residual noise and
nontrivial gradients rather than settling to zero. The 915-step spike later recedes;
the available horizon does not prove future boundedness.

The original actor has **2,181 exactly zero gradient norms out of 4,744 updates**,
first at learner update 1,267 (environment step 1,523), and a final zero gradient.
Separate TD and DDPG have no exactly zero actor norms. No action-saturation fraction,
action histogram, actor pre-tanh value, action-gradient norm, or trajectories were
logged in any screening run. The original's nearly constant return and vanishing
actor gradient are consistent with saturation, but do not distinguish saturated tanh
from a flat critic action derivative or other cancellation. No additional environment
rollout was performed to resolve this.

Alignment diagnostics were sampled 47 times. Separate TD's Q2 objective/shared-D
gradient cosine is negative at **45/47** samples, ranging from **-0.987 to 0.270**.
This is expected to be possible: detached current-input TD regression optimizes D′,
not the original shared-D bootstrap gradient. It is not evidence of a reversed optimizer
sign. Q1 alignment ranges **0.986–1.000**. Actor loss/shared-D alignment is negative
at **25/47** samples, ranging **-0.991–0.996**: descending the actor loss can raise
shared error locally. Raw gradient alignment is not a guaranteed finite Adam-step
descent statement.

## Earliest departure and likely mechanism

Learner metrics use update index `u`, while evaluation uses environment steps. In this
screening, learner update `u` occurred at environment step `u + 256`. Residual/value
snapshots are pre-update; phase gradient norms describe the corresponding subsequent
phase. Their slight within-step timing difference must not be interpreted as stale
training computation.

At the first update (environment step 257), DDPG's actor gradient is **0.211** versus
Separate TD's **0.279**; critic norms are **50.127** versus Q1/Q2 **26.217/24.816**.
This is the earliest logged numerical difference, expected from differing objectives,
half-MSE scaling, actor state inputs, and independent Q2 initialization. It is not
by itself an instability signal. Q1's pre-update outputs initially match DDPG's critic.
At update two they already differ slightly: current-input Q maxima **0.347128**
(DDPG) and **0.347122** (Separate TD).

The earliest clear, sustained mechanism departure is **critic value scale before the
first evaluation**, not an initial return collapse:

| Environment step | DDPG current Q maximum | Separate TD Q1 current maximum | Separate TD Q1/Q2 gradient norms |
| ---: | ---: | ---: | ---: |
| 300 | 13.112 | 36.908 | 143.561 / 133.411 |
| 400 | 16.096 | 285.673 | 204.567 / 214.886 |
| 500 | 21.024 | 297.898 | 236.914 / 196.695 |
| 915 | 37.381 | included in all-output maximum 1,372.814 | 13,684.043 / 10,311.163 |

At step 300, the actor gradient is still only **0.305** (DDPG **0.191**), while critic
scale is already separating. At step 915, half-D² and half-D′² peak at **2,126.250**
and **1,838.125**; actor gradient is **38.418**. The first evaluation at step 500 is
similarly poor for DDPG and Separate TD, masking the earlier critic-scale difference.

**Mechanistic hypothesis:** detached TD still uses the rapidly moving online Q2 in
the target. With negative rewards, repeated `Q ← r + 0.99 Q_next` regression can
quickly propagate negative value scale without a slow target copy. Q1 immediately
follows the changed Q2 target, and agreement keeps them close without anchoring
absolute scale. Actor changes then alter bootstrap actions and replay coverage.
DDPG's Polyak-smoothed targets move much more slowly. This explains the timing
better than a claim that the new actor first saturated: the new actor never has a
zero logged norm, and the critic departure occurs first. The early spike alone does
not establish overestimation, because negative values can also reflect faster return
propagation; the subsequent gradient spike and return reversal make overshoot a
credible hypothesis to test.

## Ranked causes and interpretation questions

Ranking is diagnostic priority based on this screening, not a statistical causal estimate.

| Rank | Candidate | Evidence and assessment |
| ---: | --- | --- |
| 1 | Critic step size with an online bootstrap | Q1/Q2 scale separates before evaluation and before a large actor gradient; the 915-step spike is orders above DDPG. Both critics use Adam at 1e-3 without a slow target. Strongest mechanism hypothesis for the remaining oscillation; not isolated yet. |
| 2 | Actor frequency/rate and feedback | Actor updates every learner step at 3e-4; later peaks at 125.054 versus DDPG 8.347, and actor/shared-error directions often conflict. Original zero gradients support a collapse concern, but Separate TD has no measured zero-gradient collapse or direct saturation metric. |
| 3 | Adam versus board gradient steps | Source derivation uses `-eta * gradient` critic steps and actor ascent. Adam uses moments/adaptive coordinates with betas 0.9/0.999; a positive raw gradient inner product does not prove descent after Adam. Explicit experimental choice, not an equation bug. |
| 4 | One seed and 5,000 steps | Separate TD improves again near the end; DDPG improves dramatically in the last 1,500 steps. The budget cannot establish asymptotic ranking or across-seed reliability. It already suffices to observe substantial early scale differences. Longer time alone need not cure them. |
| 5 | Actor next versus current replay states | Separate TD follows the next-state board reconstruction; DDPG uses current states. Both are replay-supported, but state weighting differs at time limits and under changing policies. Plausible confound; no current-state Separate TD control exists. |
| 6 | Meaning of “train separately” | Code has distinct frozen phases/optimizers but consecutive updates on one minibatch at 1:1:1 frequency. Source sections 6.2 and 7.3 explicitly permit this starting schedule and leave other schedules open. Distinct minibatches, longer critic blocks, or unequal frequencies are possible advisor interpretations, not demonstrated missing requirements. |
| 7 | Q2 should optimize D′, shared D, or agreement only; meaning of the red loss | The preserved reconstructed source's sections 3.2, 4.2, and 7.4 explicitly use shared D plus current agreement; equation images 8 and 12 corroborate this. The new requested reconstruction explicitly replaces Q2's base with detached D′ regression and retains agreement. Agreement-only Q2 is untested and unsupported by that source. The actual later red whiteboard annotation is not in the inspected artifacts; its description alone cannot establish replacement versus an analysis quantity. This semantic question needs source/advisor clarification, not an automatic code change. |
| 8 | Differentiate D′ bootstrap as a full residual | Implemented detachment exactly follows the current request. Full-residual differentiation would add `gamma * mask * J_Zn` and change update direction; it is neither mathematically equivalent nor tested here. A displayed scalar `0.5 D′ᵀD′` without detach annotations does not resolve the intended gradient. |
| 9 | Agreement signs, lambda mapping, reduction scale | Both signs are positive as confirmed. Alpha 0.1 weights Q1 next agreement; alpha-prime 0.1 weights Q2 current agreement. Each has half-batch-mean scaling. The preserved source uses alpha/alpha-prime, not an authoritative lambda1/lambda2 mapping. Equal coefficients hide a numeric swap, but code input paths agree with the supplied equations. A board vector sum would differ by batch size 64; current specification explicitly requires means. |
| 10 | Independent versus identical initialization | Independent critics are explicitly selected and confirmed by separate construction/checkpoints. Initial agreement is only about 0.0085/0.0083; large scale growth occurs later. Identical initialization is not tested and is rejected by current runner settings; no evidence identifies it as the main cause. |
| 11 | Update order, stale tensors, missing recomputation | Q2 → fresh detached Q2 forwards → Q1 → fresh actor forward is explicit and covered by reference tests. No stale-output defect found. Simultaneous gradients would constitute a different schedule and are not supported for this opt-in objective. |
| 12 | Mask, gamma, action scaling, reward or evaluation mismatch | Actual resolved settings agree: gamma 0.99, reward scale 1, no extra wrappers, tanh-to-[-2,2], clipping, terminal-zero/truncation-bootstrap policy, separate noise-free evaluation using the same fixed seeds. No mismatch found. Time-limit bootstrapping changes the value horizon relative to 200-step evaluation, but applies to all three methods. |

The [preserved research specification](source/Three_Network_DDPG_Research_Specification.docx)
is itself a working reconstruction, not direct evidence of every later board revision.
Its equation images use shared E and positive agreement; it proposes independent
critics, Adam initially, sequential recomputation, and next-state actor input, while
leaving schedules and proof claims open. Separate TD matches the explicitly requested
experimental equations, but is not identical to that older shared-error reconstruction.
This screening cannot reject the full three-network research idea or settle the advisor's
intended objective, the red annotation, or a convergence argument.

## Plot references and audit limits

Existing aggregation was read, and additional diagnostic SVGs were generated from
that saved aggregate only. All figures and raw artifacts remain ignored by Git.

These are paths to local generated artifacts, not files distributed in this repository. The paths are retained for provenance; access requires the original run artifacts.

- Shared-scale return comparison: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/comparison-evaluation_return_mean.svg`
- Separate TD shared error: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-shared_squared_error.svg`
- Original shared error: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-shared-error-shared_squared_error.svg`
- Separate TD own error: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-q2_td_squared_error.svg`
- Original own error: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-shared-error-q2_td_squared_error.svg`
- Separate TD current agreement: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-current_agreement_error.svg`
- Separate TD next agreement: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-next_agreement_error.svg`
- Separate TD Q scale: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-q_abs_max.svg`
- Original Q scale: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-shared-error-q_abs_max.svg`
- Separate TD actor gradients: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-actor_gradient_norm.svg`
- Separate TD Q1 gradients: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-critic_one_gradient_norm.svg`
- Separate TD Q2 gradients: `runs/pendulum-separate-td-screening-01/pendulum-separate-td-screening-01/reports/plots/three-network-separate-td-critic_two_gradient_norm.svg`

Only the return comparison has a shared plotting scale across variants; mechanism
panels auto-scale separately. Compare numeric tables or axis labels, not visual panel
heights. There is also an existing diagnostic-axis limitation: aggregation preserves
the raw metric `step`, so diagnostic curves begin at learner index 1, whereas their
plot x-axis is labeled environment steps. For this screening, add 256 to diagnostic
indices when comparing them with the evaluation curve. Evaluation returns already
use environment steps, so their AUC and timing are unaffected. The tables in this
report apply that conversion explicitly. This is a reporting issue rather than an
objective violation and was not changed in the algorithm audit. No trajectories,
phase-by-phase held-out residuals, direct action-saturation metrics, or calibration
targets were available. There were no additional seeds, longer Separate TD runs,
alternative batches/ratios, identical initialization, SGD, full-residual gradients,
agreement-only Q2, guards, beta terms, or target-network variants in this screening.

## Recommended minimal next experiment: common critic learning rate

Run **two** Separate TD variants: an unchanged `1e-3` critic-rate reference and a
`1e-4` critic-rate treatment. Change both critics' rates together as one common-rate
factor. Keep the actor at 3e-4, both agreement coefficients at 0.1, seed 11, 5,000
steps, evaluation seeds 1001–1005, all architecture/replay/noise/Adam settings, next-state
actor objective, and sequential order unchanged. This tests the earliest observed
critic-scale departure without changing the residual definition or adding targets.

Estimated workload: **2 runs**, **10,000 training environment steps**, **9,488 learner
updates**, **100 evaluation episodes / 20,000 evaluation steps**, and **30,000 total
environment interactions**. One worker runs sequentially. The previous Separate TD
run took 726 seconds including evaluations; two runs of comparable per-step cost
suggest roughly 24 minutes on the same idle CPU, but this is an extrapolation, not a
runtime guarantee. Use the completed reference's measured wall time to revise it.

Inspect the first 1,000 steps for Q scale, both residual spikes and critic norms, then
the whole return curve, normalized AUC, last-five mean, and final episode returns.
Do not select a candidate by one final checkpoint. A reduced early spike accompanied
by slower return learning is a time-scale tradeoff, not automatic success; poor early
fit alone may also reflect the smaller rate. If scale/error spikes persist, spending a
longer budget is not justified merely by finite completion. Any promising result still
needs paired development-seed replication before a scientific ranking.

From the repository root, the commands below create a small manifest under the ignored `runs/manual-plans`
directory, preserving the source manifest and prior results. They are manual commands;
no run or resume command was executed during this audit. Do not regenerate or edit
the manifest after its study starts. Keep the code commit fixed for run/resume.

```powershell
conda activate three-network-ddpg
git status --short --branch
git rev-parse HEAD
$Manifest = "runs/manual-plans/pendulum-separate-td-critic-rate-01.yaml"
@'
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
source = OmegaConf.to_container(OmegaConf.load("experiments/manifests/pendulum_separate_td_screening_01.yaml"), resolve=True)
reference = deepcopy(next(v for v in source["variants"] if v["id"] == "three-network-separate-td"))
reference["id"] = "separate-td-critic-lr-1e-3"
slower = deepcopy(reference)
slower["id"] = "separate-td-critic-lr-1e-4"
slower["overrides"] = ["algorithm.critic_one_learning_rate=0.0001" if x.startswith("algorithm.critic_one_learning_rate=") else "algorithm.critic_two_learning_rate=0.0001" if x.startswith("algorithm.critic_two_learning_rate=") else x for x in slower["overrides"]]
source["name"] = "pendulum-separate-td-critic-rate-01"
source["description"] = "Manual development test of one common critic-rate factor; no convergence claim."
source["variants"] = [reference, slower]
source["execution"]["output_root"] = "runs/pendulum-separate-td-critic-rate-01"
source["preregistration"]["hypothesis"] = "A smaller common critic step size may reduce the early online-bootstrap spike; inspect the full return trajectory for slower learning."
source["preregistration"]["fixed_factors"] = [x for x in source["preregistration"]["fixed_factors"] if x != "learning rates"] + ["actor learning rate", "critic-rate ratio"]
source["preregistration"]["varied_factors"] = ["common critic learning rate"]
destination = Path("runs/manual-plans/pendulum-separate-td-critic-rate-01.yaml")
if destination.exists():
    raise SystemExit("Manifest already exists; inspect and reuse it rather than overwrite it.")
destination.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(OmegaConf.create(source), destination)
'@ | python -B -
three-network-ddpg experiment validate $Manifest
three-network-ddpg experiment plan $Manifest
```

Inspect the plan: exactly two runs, seed 11, 5,000 steps each, and only the two critic
rates differing. Then launch manually and analyze after completion:

```powershell
three-network-ddpg experiment run $Manifest
three-network-ddpg experiment status $Manifest
# Only if interrupted or incomplete, after the previous process has stopped:
three-network-ddpg experiment resume $Manifest
three-network-ddpg experiment aggregate $Manifest
three-network-ddpg experiment plot $Manifest --metric evaluation_return_mean
three-network-ddpg experiment plot $Manifest --metric shared_squared_error
three-network-ddpg experiment plot $Manifest --metric q2_td_squared_error
three-network-ddpg experiment plot $Manifest --metric current_agreement_error
three-network-ddpg experiment plot $Manifest --metric next_agreement_error
three-network-ddpg experiment plot $Manifest --metric q_abs_max
three-network-ddpg experiment plot $Manifest --metric actor_gradient_norm
three-network-ddpg experiment plot $Manifest --metric critic_one_gradient_norm
three-network-ddpg experiment plot $Manifest --metric critic_two_gradient_norm
```

`run` starts only never-started runs; `resume` retries eligible incomplete attempts,
not an exact continuation of replay state. Ctrl+C records an interrupted attempt.
Output root is
`runs/pendulum-separate-td-critic-rate-01/pendulum-separate-td-critic-rate-01`, with
per-run resolved configs, per-attempt metrics/checkpoints/markers, and `reports/aggregate.json`
and `reports/plots`. Preserve all statuses and attempt records when interpreting this
single-factor development experiment.
