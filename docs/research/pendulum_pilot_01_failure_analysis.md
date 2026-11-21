# Pendulum Pilot 01 Failure Analysis

## Executive conclusion

Pendulum Pilot 01 executed 24 planned logical runs: three Stage A pipeline-smoke runs, nine Stage B mechanism-pilot runs, and twelve Stage C agreement-pilot runs. All 24 ultimately completed, and none was classified by the runner as failed, divergent, non-finite, or incomplete. One Stage C attempt (`three-network-current`, seed 33) was interrupted by the operator after learner step 10,546 and was then rerun as attempt 2 to completion. Thus, "failure" in this report means failure to produce stable value learning or useful control under the tested setting, not a process-status failure.

The two DDPG controls learned useful Pendulum policies in Stage B. Standard DDPG achieved normalized AUC `-324.911 +/- 25.915` and a last-five-evaluations aggregate return of `-195.529 +/- 4.642`; DDPG without a target actor achieved `-333.444 +/- 42.804` and `-197.369 +/- 4.155`, respectively. Removing only the target actor therefore did not materially damage this pilot.

The target-free three-network shared-error configuration did not learn useful control. Its Stage B normalized AUC was `-1418.546 +/- 47.925`, while its critics' absolute output scale, agreement errors, shared residual, and gradient norms grew by many orders of magnitude. Stage C reproduced the same failure for agreement mode `none` and found that coefficients of 0.1 for `current`, `next`, or `both` agreement did not stabilize learning. The `next` and `both` modes produced the largest value explosions; `current` was numerically less extreme but still unstable and did not approach the DDPG controls.

These observations reject only the tested target-free, sequential, next-state-actor settings at the stated learning rates and agreement coefficients. They do not establish that the full three-network idea fails.

## Evidence and interpretation boundaries

This analysis uses the three manifests, each run's locked run specification and resolved configuration, append-only metrics, completion markers, summaries, aggregate reports, generated SVG plots, and the implementation at the recorded Git commits. Stage A was executed at `f9ac6c3967301b71e9e5f930074ac5395b67beb0`; Stages B and C were executed at `4218aa3e66b0c38e9fe86636e3a4981d0c4356ab`. All used CPython 3.11.16, PyTorch 2.14.0, Gymnasium 1.3.0, NumPy 2.4.6, CPU execution, and one worker.

The return column below is the mean over the five fixed evaluation episodes at the final checkpoint. Normalized AUC is trapezoidal area divided by the covered environment-step interval. The aggregate report's "final return" is instead the per-run mean over the configured last `k` evaluation checkpoints (`k=4` in Stage A and `k=5` in Stages B and C), summarized across seeds. Standard deviations are sample standard deviations. Confidence intervals are the protocol's deterministic 2,000-sample, 95% bootstrap intervals; with one seed, no interval is reported.

Stage A is a pipeline smoke, not scientific evidence. Stages B and C are bounded development pilots, not confirmatory experiments. Training seeds `11, 22, 33` are separate from fixed evaluation seeds `1001, 1002, 1003, 1004, 1005`.

## Executed study matrix

| Stage | Purpose | Variants | Training seeds | Steps per run | Logical runs | Evaluation schedule |
|---|---|---|---|---:|---:|---|
| A | Pipeline smoke | DDPG; DDPG without target actor; three-network shared-error-only | 11 | 2,000 | 3 | Every 500 steps, 5 fixed episodes |
| B | Mechanism pilot | Same three variants | 11, 22, 33 | 20,000 | 9 | Every 1,000 steps, 5 fixed episodes |
| C | Agreement pilot | Three-network `none`, `current`, `next`, `both` | 11, 22, 33 | 20,000 | 12 | Every 1,000 steps, 5 fixed episodes |

The completed runs contain 426,000 training environment steps. With 256 warm-up steps per run, Stage A performed 1,744 learner updates per run and Stages B/C performed 19,744 per run. The completed attempts consumed 12.49 hours of measured serial training/evaluation wall time. This excludes the interrupted Stage C attempt.

### Agreement coefficients in Stage C

| Variant | Agreement mode | `alpha` for next-input agreement | `alpha_prime` for current-input agreement |
|---|---|---:|---:|
| `three-network-none` | none | 0.0 | 0.0 |
| `three-network-current` | current | 0.0 | 0.1 |
| `three-network-next` | next | 0.1 | 0.0 |
| `three-network-both` | both | 0.1 | 0.1 |

No alpha sweep was run. Beta regularization was disabled in every three-network run.

## Environment, architecture, and parameterization

### Environment and interfaces

All runs used unwrapped `Pendulum-v1`. Observation normalization was disabled and reward scale was 1.0. The observation is the three-dimensional Pendulum state supplied by Gymnasium. The action is one continuous torque. The actor maps 3 inputs to 1 tanh output and rescales it to the environment bounds `[-2, 2]`. A critic concatenates the 3-vector observation and 1-vector environment-scale action, so its input width is 4 and its output is one scalar Q estimate.

Every actor and critic is an MLP with widths `128, 128`, ReLU after each hidden linear layer, and a linear scalar critic output. The actor adds tanh and environment action rescaling. The architecture is therefore:

```text
Actor:  observation(3) -> Linear(3,128) -> ReLU
                        -> Linear(128,128) -> ReLU
                        -> Linear(128,1) -> tanh -> scale to [-2,2]

Critic: concat(observation(3), action(1))
        -> Linear(4,128) -> ReLU
        -> Linear(128,128) -> ReLU
        -> Linear(128,1)
```

The exact trainable parameter counts are:

| Network | Calculation | Parameters |
|---|---|---:|
| Actor | `(3x128+128) + (128x128+128) + (128x1+1)` | 17,153 |
| One critic | `(4x128+128) + (128x128+128) + (128x1+1)` | 17,281 |
| DDPG online actor + critic | `17,153 + 17,281` | 34,434 trainable |
| DDPG targets | copied actor + copied critic | 34,434 frozen |
| DDPG total network parameters | online + targets | 68,868 |
| DDPG without target actor | online actor + online critic + target critic | 51,715 total; 34,434 trainable |
| Three-network | actor + Critic One + Critic Two | 51,715, all trainable |

The summaries record `parameter_count` values of 137,752 (DDPG), 120,597 (DDPG without target actor), and 155,165 (three-network). Those values are not model-only parameter counts: the reporting helper recursively counts every tensor in `algorithm.state_dict()` after training, including Adam optimizer-state tensors. The table above is the model architecture count.

### Initialization and optimizers

Each algorithm calls `torch.manual_seed(training_seed)` immediately before constructing its networks. No custom layer initializer is applied. PyTorch `nn.Linear` defaults are therefore used: weights use the module's default Kaiming-uniform initialization and biases use the corresponding fan-in uniform bound. The two proposed critics are constructed separately from successive RNG draws and are independent, not identical.

Standard DDPG creates its target actor and target critic by exact `deepcopy` of the online networks and freezes their parameters. DDPG without a target actor omits only the target actor; it retains an exact-copy frozen target critic. The three-network implementation has no target actor and no target critic.

The trainable networks use separate Adam optimizers with default betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, and no AMSGrad. Actor learning rate is `3e-4`; every critic learning rate is `1e-3`. Gradient clipping is disabled.

## Exact training and evaluation settings

The following settings were fixed across comparable variants:

| Setting | Value |
|---|---|
| Discount `gamma` | 0.99 |
| Replay capacity | 10,000 transitions, CPU ring buffer |
| Replay batch | 64, sampled without replacement with a seeded NumPy generator |
| Warm-up | First 256 actions sampled from the environment action space |
| Learner cadence | One update after every post-warm-up environment step |
| Exploration | Independent Gaussian action noise, standard deviation 0.1, no noise clipping |
| Action handling | Actor output rescaled to environment bounds; final action clipped to bounds |
| Actor update delay | 1 (actor updated on every learner step) |
| Observation normalization | Disabled |
| Reward scaling | 1.0 |
| Device and workers | CPU, one sequential worker |
| Episode boundary | `terminated` and `truncated` stored separately |
| Bootstrap policy | Bootstrap across truncation; mask is zero only for true termination |
| Tracking | JSONL and CSV enabled; TensorBoard and W&B disabled |
| Gradient diagnostics | Every 100 learner updates for three-network variants |
| Checkpoint | Final checkpoint written after final evaluation |

Pendulum episodes reached the 200-step time limit. Because `bootstrap_on_truncation=true`, those truncations retained a bootstrap mask of one. Evaluation disabled exploration and used exactly seeds `1001` through `1005`. Stage A made four evaluations per run; Stages B and C made twenty. Each evaluation consisted of five 200-step episodes.

### DDPG update order

For standard DDPG, the detached target is

$$
y_{\mathrm{DDPG}} = r + \gamma m\,Q_{\bar\phi}\!\left(s',\mu_{\bar\theta}(s')\right).
$$

The critic minimizes PyTorch mean squared error, `MSE(Q_phi(s,a), y_DDPG)`. The actor then minimizes

$$
L_{\mathrm{actor,DDPG}}=-\mathbb{E}\left[Q_\phi(s,\mu_\theta(s))\right]
$$

with critic parameters frozen but the critic action path connected to the actor. Finally, target parameters receive a Polyak update with `tau=0.005`.

DDPG without a target actor uses the same update order and target critic, but substitutes the detached online action `mu_theta(s')` for `mu_bar_theta(s')` in the TD target. This is the only intended algorithmic difference between the two controls.

## Three-network equations and gradient paths

For a replay batch, define a detached next action and four critic outputs:

$$
a'=\operatorname{stopgrad}(\mu_\theta(s')),\quad
Y=Q_{\phi}(s,a),\quad
Y_1=Q_{\phi}(s',a'),\quad
Z_1=Q_{\phi'}(s,a),\quad
Z=Q_{\phi'}(s',a').
$$

The shared temporal-difference residual and monitored shared squared error are

$$
\delta=r+\gamma m Z-Y,
\qquad
e=\frac{1}{2}\mathbb{E}[\delta^2].
$$

The agreement errors are

$$
c_{\mathrm{current}}=\frac{1}{2}\mathbb{E}[(Z_1-Y)^2],
\qquad
c_{\mathrm{next}}=\frac{1}{2}\mathbb{E}[(Y_1-Z)^2].
$$

In the sequential schedule used by every pilot run, updates occur in this exact order:

1. **Critic Two.** Actor outputs and Critic One's `Y` are detached. Critic Two minimizes

   $$
   L_2=e+\alpha'c_{\mathrm{current}}.
   $$

   Only parameters `phi_prime` receive gradients.

2. **Critic One.** Next actions are recomputed, and the newly updated Critic Two's `Z` is detached. Critic One minimizes

   $$
   L_1=e+\alpha c_{\mathrm{next}}+\beta R_\beta.
   $$

   Every pilot used `beta=0` and `R_beta=0`. Only parameters `phi` receive gradients.

3. **Actor.** Both critics' parameters are frozen. The actor minimizes

   $$
   L_\theta=-\mathbb{E}\left[Q_{\phi'}(s',\mu_\theta(s'))\right].
   $$

   The Critic Two output is not detached from its action input, so gradients pass through Critic Two to the actor while neither critic accumulates parameter gradients.

The next-state actor objective differs from the DDPG controls' current-state objective. Validation batches were disabled; no guard or acceptance rule changed any update. The logged validation/diagnostic machinery remained monitoring-only.

### Variant differences

| Variant | Actor bootstrap source | Bootstrap critic | Target networks | Critic system | Agreement |
|---|---|---|---|---|---|
| DDPG | frozen target actor | frozen target critic | actor and critic | one online critic | n/a |
| DDPG without target actor | detached online actor | frozen target critic | critic only | one online critic | n/a |
| Three-network shared-error / `none` | detached online actor | trainable Critic Two | none | two independent trainable critics | none |
| Three-network `current` | same as `none` | same | none | same | Critic Two current-input penalty, `alpha_prime=0.1` |
| Three-network `next` | same as `none` | same | none | same | Critic One next-input penalty, `alpha=0.1` |
| Three-network `both` | same as `none` | same | none | same | both penalties at 0.1 |

## Numerical results

### Stage A pipeline smoke

Stage A exercised artifact creation and was not sized to assess learning.

| Variant | Seed | Final evaluation return | Normalized AUC | Wall time (s) |
|---|---:|---:|---:|---:|
| DDPG | 11 | -1265.828 | -1455.806 | 58.064 |
| DDPG without target actor | 11 | -1232.475 | -1457.174 | 57.662 |
| Three-network shared-error-only | 11 | -1403.235 | -1389.554 | 228.535 |

The one-seed last-four-evaluation aggregate returns were `-1469.281`, `-1468.517`, and `-1389.810`, respectively. No confidence interval is defined for one seed.

Despite completing, the three-network run already showed mechanism instability: shared error ended at `10.950` after peaking at `1,441.301`; current agreement ended at `17,315.016` (peak `56,274.332`); next agreement ended at `461.343` (peak `6,058.071`); Q absolute mean ended at `234.106` and Q absolute maximum at `907.849` (peak `1,327.102`). Actor, Critic One, and Critic Two gradient norms peaked at `835.880`, `6,105.831`, and `7,463.721`. The DDPG controls' Q absolute maxima ended near 72 and their gradient peaks were below 67 for the critic and 1.60 for the actor.

Plot references:

These are paths to local generated artifacts, not files distributed in this repository. The paths are retained for provenance; access requires the original run artifacts.

- Stage A comparison: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-a/reports/plots/comparison-evaluation_return_mean.svg`
- Stage A DDPG: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-a/reports/plots/ddpg-pipeline-smoke-evaluation_return_mean.svg`
- Stage A no-target-actor DDPG: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-a/reports/plots/ddpg-no-target-actor-pipeline-smoke-evaluation_return_mean.svg`
- Stage A three-network shared-error: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-a/reports/plots/three-network-shared-error-pipeline-smoke-evaluation_return_mean.svg`

### Stage B per-seed control results

| Variant | Seed | Final evaluation return | Normalized AUC | Wall time (s) |
|---|---:|---:|---:|---:|
| DDPG | 11 | -191.992 | -353.363 | 617.016 |
| DDPG | 22 | -194.429 | -302.657 | 604.900 |
| DDPG | 33 | -200.055 | -318.713 | 624.215 |
| DDPG without target actor | 11 | -197.100 | -367.690 | 691.847 |
| DDPG without target actor | 22 | -192.024 | -285.457 | 659.767 |
| DDPG without target actor | 33 | -194.982 | -347.185 | 649.727 |
| Three-network shared-error-only | 11 | -1403.134 | -1386.430 | 2717.282 |
| Three-network shared-error-only | 22 | -1403.138 | -1395.576 | 2685.073 |
| Three-network shared-error-only | 33 | -1403.134 | -1473.632 | 2719.177 |

### Stage B aggregates

| Variant | Last-five return mean +/- SD [95% CI] | AUC mean +/- SD [95% CI] | Mean runtime (s) |
|---|---|---|---:|
| DDPG | -195.529 +/- 4.642 `[-200.661, -191.624]` | -324.911 +/- 25.915 `[-353.363, -302.657]` | 615.377 |
| DDPG without target actor | -197.369 +/- 4.155 `[-201.717, -193.438]` | -333.444 +/- 42.804 `[-367.690, -285.457]` | 667.114 |
| Three-network shared-error-only | -1399.261 +/- 6.678 `[-1403.135, -1391.550]` | -1418.546 +/- 47.925 `[-1473.632, -1386.430]` | 2707.177 |

Both DDPG controls improved sharply between 1,000 and 5,000 steps and finished near `-195`. The three-network policies remained near a poor fixed-control regime. Their almost identical final returns across training seeds do not demonstrate convergence; together with the exploding Q values and occasional zero actor gradients, they are consistent with policy saturation or collapse.

### Stage B mechanism and scale metrics

Each cell below is `mean final value across seeds / largest value observed in any seed`.

| Variant | Shared error | Current agreement | Next agreement | Q absolute mean | Q absolute max |
|---|---:|---:|---:|---:|---:|
| DDPG | n/a | n/a | n/a | not logged | 132.334 / 265.329 |
| DDPG without target actor | n/a | n/a | n/a | not logged | 134.701 / 262.376 |
| Three-network shared-error-only | 72,203.467 / 38,879,812 | 19,063,196 / 233,596,800 | 1,702,071.990 / 176,000,976 | 19,024.561 / 60,469.664 | 50,829.290 / 107,953.344 |

| Variant | Actor gradient norm | Critic / Critic One gradient norm | Critic Two gradient norm |
|---|---:|---:|---:|
| DDPG | 2.314 / 24.873 | 69.069 / 993.388 | n/a |
| DDPG without target actor | 4.111 / 25.980 | 113.585 / 1,945.051 | n/a |
| Three-network shared-error-only | 0.054 / 70,228.672 | 219,006.625 / 47,019,652 | 180,199.297 / 42,700,232 |

For agreement mode `none`, critic-objective/shared-error gradient cosines remain approximately one by construction, because each critic objective is only its shared-error term. That local alignment did not prevent moving-target instability over repeated sequential updates. Actor/shared-error cosine ranged close to both -1 and +1, confirming that policy improvement is not consistently a descent direction for the shared error.

Plot references:

- Stage B comparison: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-b/reports/plots/comparison-evaluation_return_mean.svg`
- Stage B DDPG: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-b/reports/plots/ddpg-evaluation_return_mean.svg`
- Stage B no-target-actor DDPG: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-b/reports/plots/ddpg-no-target-actor-evaluation_return_mean.svg`
- Stage B three-network shared-error: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-b/reports/plots/three-network-shared-error-evaluation_return_mean.svg`

### Stage C per-seed agreement results

| Agreement mode | Seed | Final evaluation return | Normalized AUC | Wall time (s) |
|---|---:|---:|---:|---:|
| none | 11 | -1403.134 | -1386.430 | 2603.210 |
| none | 22 | -1403.138 | -1395.576 | 2843.352 |
| none | 33 | -1403.134 | -1473.632 | 2613.024 |
| current | 11 | -1262.530 | -1412.704 | 2773.970 |
| current | 22 | -1592.308 | -1507.243 | 2873.582 |
| current | 33 | -1474.255 | -1492.323 | 2736.342 |
| next | 11 | -1374.011 | -1374.011 | 2963.014 |
| next | 22 | -1599.184 | -1597.887 | 2646.079 |
| next | 33 | -1374.011 | -1374.011 | 2615.505 |
| both | 11 | -1374.011 | -1374.011 | 2574.081 |
| both | 22 | -1403.134 | -1418.463 | 2716.759 |
| both | 33 | -1374.011 | -1300.783 | 2706.546 |

### Stage C aggregates

| Mode | Last-five return mean +/- SD [95% CI] | AUC mean +/- SD [95% CI] | Mean runtime (s) |
|---|---|---|---:|
| none | -1399.261 +/- 6.678 `[-1403.135, -1391.550]` | -1418.546 +/- 47.925 `[-1473.632, -1386.430]` | 2686.529 |
| current | -1487.901 +/- 119.748 `[-1593.414, -1357.750]` | -1470.756 +/- 50.826 `[-1507.243, -1412.704]` | 2794.631 |
| next | -1448.812 +/- 129.560 `[-1598.415, -1374.011]` | -1448.636 +/- 129.255 `[-1597.887, -1374.011]` | 2741.533 |
| both | -1383.718 +/- 16.814 `[-1403.134, -1374.011]` | -1364.419 +/- 59.424 `[-1418.463, -1300.783]` | 2665.795 |

The apparent AUC ordering within Stage C must not be used to declare a winner. All four modes remained far below the Stage B controls, all developed extreme critic scales, and the sample contains only three development seeds. In particular, `both` had the best mean AUC among the four while also producing multi-million Q magnitudes and billion-scale agreement errors.

### Stage C mechanism and scale metrics

Each cell is `mean final value across seeds / largest value observed in any seed`.

| Mode | Shared error | Current agreement | Next agreement | Q absolute mean | Q absolute max |
|---|---:|---:|---:|---:|---:|
| none | 72,203 / 38,879,812 | 19,063,196 / 233,596,800 | 1,702,072 / 176,000,976 | 19,025 / 60,470 | 50,829 / 107,953 |
| current | 100,359 / 81,090,768 | 1,189,568 / 249,337,248 | 1,209,116 / 58,520,184 | 18,088 / 83,178 | 40,001 / 138,736 |
| next | 257,957,019 / 1,468,501,632 | 930,649,120 / 4,530,878,976 | 819,955,989 / 3,155,293,696 | 2,135,810 / 3,987,295 | 2,864,358 / 4,950,832 |
| both | 44,978,752 / 900,408,576 | 725,333,212 / 5,264,294,912 | 712,328,966 / 4,326,915,072 | 2,089,287 / 5,344,545 | 2,328,794 / 5,828,528 |

| Mode | Actor gradient norm | Critic One gradient norm | Critic Two gradient norm |
|---|---:|---:|---:|
| none | 0.054 / 70,228.672 | 219,006.625 / 47,019,652 | 180,199.297 / 42,700,232 |
| current | 385.810 / 244,611.656 | 305,818.995 / 113,983,880 | 143,935.308 / 97,272,616 |
| next | 0.000 / 61,981.504 | 207,867,315 / 2,222,095,104 | 201,506,587 / 2,031,491,456 |
| both | 0.000 / 105,353.820 | 194,671,630 / 1,803,511,680 | 326,588,082 / 2,193,529,600 |

The zero final actor gradient norms for all `next` and `both` seeds, after very large earlier peaks, are consistent with saturated actor/critic pathways. This is measured at the final logged learner update; the saturation explanation is a hypothesis, not a proven cause.

Plot references:

- Stage C comparison: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-c/reports/plots/comparison-evaluation_return_mean.svg`
- Stage C none: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-c/reports/plots/three-network-none-evaluation_return_mean.svg`
- Stage C current: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-c/reports/plots/three-network-current-evaluation_return_mean.svg`
- Stage C next: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-c/reports/plots/three-network-next-evaluation_return_mean.svg`
- Stage C both: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-c/reports/plots/three-network-both-evaluation_return_mean.svg`

## Reliability and failure record

| Stage | Planned | Completed | Failed | Divergent | Non-finite | Current interrupted/incomplete |
|---|---:|---:|---:|---:|---:|---:|
| A | 3 | 3 | 0 | 0 | 0 | 0 |
| B | 9 | 9 | 0 | 0 | 0 | 0 |
| C | 12 | 12 | 0 | 0 | 0 | 0 |

The runner's reliability classification did not flag the finite but enormous values as divergent. This is an important distinction: the three-network runs were mechanically complete and finite, yet scientifically unstable by their own mechanism diagnostics.

The only interrupted attempt was Stage C `three-network-current`, seed 33, attempt 1. Its marker states `interrupted by operator`; metrics extend through learner step 10,546. Attempt 2 ran from a clean attempt directory and completed all 20,000 environment steps. This is an operational interruption, not evidence of an algorithm exception or non-finite failure.

## What the evidence establishes

### Measured findings

1. Standard DDPG and DDPG without a target actor both learned useful Pendulum control under the shared training budget and paired seeds. Their aggregate returns and AUC overlap substantially.
2. The target-free three-network shared-error-only model did not learn useful control under the tested sequential update. Stage B and Stage C `none` reproduce each other at the metric level.
3. Three-network value magnitudes, shared errors, agreement errors, and critic gradients increased by orders of magnitude while remaining finite.
4. A coefficient of 0.1 on current-input agreement reduced the final value scale relative to `next` and `both`, but did not stabilize control or keep errors bounded.
5. Next-input agreement and two-sided agreement produced the largest Q scales and billion-scale agreement/gradient measurements. Agreement alone did not anchor the absolute value scale.
6. Lower shared TD error did not imply a better policy. The short Stage A shared-error run had a superficially competitive AUC while already showing rapidly growing disagreement and Q scale; the longer runs exposed the failure.
7. Three-network runs were roughly four times slower than the DDPG controls on this CPU. This is consistent with training a second critic and computing extensive per-update diagnostics, but the pilot did not isolate runtime causes.

### Hypotheses, not demonstrated causes

1. **Moving-target positive feedback.** Critic Two supplies a trainable bootstrap target to Critic One, while Critic One supplies a trainable target to Critic Two. Sequentially updating both at every step may amplify value scale without a slow target anchor.
2. **Agreement without absolute anchoring.** Squared agreement can make two critics agree while both drift. Stage C supports this concern because `next` and `both` had extreme agreement and Q scales despite explicit penalties.
3. **Actor/critic time-scale mismatch.** The actor updates every learner step and maximizes the same Critic Two that supplies the bootstrap. Large early actor gradients followed by zero gradients in `next` and `both` may indicate saturation after runaway critic values.
4. **Next-state actor objective.** The proposed actor uses `s'`, while the successful DDPG controls use replay states `s`. This is a confound in the pilot and could change both state weighting and feedback through Critic Two.
5. **Learning rates are too aggressive for a target-free coupled system.** Rates borrowed from the stable DDPG control may be inappropriate when both sides of the Bellman residual move every update.
6. **Lack of explicit Q-scale control.** Beta regularization was disabled. The original linear beta proposal is unbounded in the wrong direction for negative values, but the implemented `l2_y` alternative was not tested.

The pilot does not isolate these causes. Multiple mechanisms may interact.

## Match to the original research specification

The implemented proposed model matches the reconstructed specification in the following material respects:

- one deterministic actor and two separately instantiated, independently optimized critics;
- no target actor and no target critic in the proposed main method;
- Critic One on the current side and Critic Two on the next-state bootstrap side of the shared Bellman residual;
- current-input and next-input agreement terms with separate `alpha_prime` and `alpha` coefficients;
- next-state actor objective through Critic Two;
- actor detached during critic updates;
- the other critic detached during each critic's update;
- actor gradients retained through Critic Two's action input while critic parameters are frozen;
- sequential order Critic Two, Critic One, then actor, with recomputation after the earlier update;
- separate optimizers and diagnostics for losses, error, agreement, Q scale, gradients, return, and cost;
- explicit truncation-aware bootstrap masking and environment action rescaling.

The pilot is not a test of the full research program described in the specification. It tests only one sequential schedule, independent initialization, next-state actor input, actor update frequency one, no guard, no beta regularizer, one network width, one set of learning rates, and one environment. The specification explicitly treats schedule, guard, beta form, actor input, initialization, target-network ablations, and time-scale choices as open research decisions. No convergence theorem was tested or established.

## What has not been tested

- simultaneous updates from one parameter snapshot;
- guarded or backtracking updates, whether joint, critic-only, or actor-only;
- current-state actor objective in the three-network model;
- delayed or less frequent actor updates;
- smaller actor or critic learning rates and alternative critic-to-actor update ratios;
- `l2_y` Q-scale regularization; the rejected `l2_y_z` mode remains unavailable;
- any beta-coefficient study;
- smaller, scheduled, or adaptive agreement coefficients;
- identical critic initialization;
- explicit gradient clipping as a named ablation;
- a slowly updated target critic or other stabilizing anchor as a controlled ablation;
- observation or reward normalization;
- other network widths, replay sizes, batch sizes, noise scales, or warm-up lengths;
- held-out confirmatory seeds, additional environments, GPU execution, or multiple independent machines;
- statistical claims beyond this three-seed development pilot;
- global convergence, optimal-policy convergence, monotonic return improvement, or novelty.

## Ranked changes for the next experiment

These are proposed controlled changes, not fixes implemented by this report.

1. **Add a stabilizing bootstrap control as a named ablation.** Compare the current target-free Critic Two against a slowly updated target critic or Polyak-smoothed Critic Two while preserving all other factors. The largest measured pathology is an unanchored value scale, so this directly tests moving-target feedback.
2. **Enable the existing guard/backtracking mechanism in a narrowly scoped pilot.** Start with critic-phase or actor-phase acceptance and retain a separate validation batch as monitoring-only. Record rejection rate, accepted learning rate, and pre/post shared error. This tests whether finite-step increases can be controlled without claiming policy improvement from lower error.
3. **Decouple the actor time scale.** Test a lower actor learning rate and delayed actor updates, one factor at a time. The actor should not update on every critic step until the critic scale is demonstrably bounded.
4. **Ablate current-state versus next-state actor input.** The successful DDPG controls use current replay states, while all failed proposed runs use next states. This removes a major confound without changing the critic residual.
5. **Introduce explicit one-sided Q-scale control.** Test the implemented `l2_y` regularizer with preregistered coefficients. Do not use `l2_y_z`, and do not reuse the original unbounded linear beta term as if it were a norm bound.
6. **Reduce critic step size before adding stronger agreement.** Run a small, bounded learning-rate comparison with agreement disabled. Agreement coefficients should only be revisited after the shared-error-only system no longer exhibits runaway Q values.
7. **Retest agreement conservatively.** If the reference is stable, compare smaller or scheduled current-only coefficients first. Stage C gives no support for proceeding directly with `next=0.1` or `both=0.1`.
8. **Compare sequential and simultaneous critic updates.** Simultaneous gradients from a common snapshot may reduce within-step target chasing, although it cannot by itself guarantee stability.
9. **Use gradient clipping only as an explicit diagnostic ablation.** It may prevent catastrophic steps but can hide the underlying scale problem; report clipped-step frequency and pre-clip norms.

The next pilot should keep development and confirmatory seeds separate, use paired seeds, retain the DDPG controls, and require bounded Q scale and finite gradients as a stage gate before spending a longer budget. A variant should not be selected from its best single seed or from shared-error reduction alone.

## Artifact locations

- Manifests: `experiments/manifests/pendulum_pilot_01_stage_{a,b,c}.yaml`
- Stage roots: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-{a,b,c}`
- Per-run resolved configuration: `runs/<stage>/runs/<run-id>/resolved_config.yaml`
- Per-attempt metrics and summaries: `runs/<stage>/runs/<run-id>/attempts/<attempt>/`
- Aggregate reports: `runs/<stage>/reports/aggregate.json`
- Plot indexes and SVG figures: `runs/<stage>/reports/plots/`

The generated run directories are intentionally ignored by Git. This report records their results without adding raw runs, checkpoints, aggregate files, or plots to version control.
