# Current implementation and entry points

This guide describes the latest implemented research direction on `main`,
`three_network_shared_d_conditioned_paired`. "Latest" identifies the implementation,
not a demonstrated best-performing method. Standard DDPG remains the reference
baseline. No confirmed performance advantage or convergence result is claimed.

## Implementations

| Configuration name | Role |
| --- | --- |
| `ddpg` | Standard actor, critic, target actor, and target critic baseline |
| `ddpg_no_target_actor` | Online actor and critic with a target critic |
| `three_network_ddpg` | Original target-free core; shared-error or opt-in separate-TD objectives |
| `three_network_shared_d*` | Named shared-error reconstructions with their own update contracts |
| `three_network_shared_d_conditioned_paired` | Latest conditioned, paired-row reconstruction |
| `td3`, `sac`, `a2c`, `a3c` | Configuration placeholders; training is not implemented |

All feature-branch heads reviewed on 2026-11-21 were ancestors of `main`.
The implementation source of truth is the [registry](../../src/three_network_ddpg/algorithms/registry.py)
and [packaged configuration groups](../../src/three_network_ddpg/conf/algorithm/).
Earlier variants remain available for reproducibility.

## Conditioned-paired update

The implementation uses one deterministic actor and two independently initialized
critics with separate Adam optimizers. There are no target networks or beta
regularization. One ordered replay minibatch supplies every current/next input pair.

For a transition batch, let `sg` denote stop-gradient and define:

```text
a_next = sg(actor(s_next))
Y      = Q1(s, a)
Y_next = Q1(s_next, a_next)
Z_cur  = Q2(s, a)
Z_next = Q2(s_next, a_next)
D      = r + gamma * bootstrap_mask * Z_next - Y
```

Each learner iteration performs these phases:

1. Update Q2 using `0.5 * mean(D**2) + alpha_prime * 0.5 * mean((Z_cur - sg(Y))**2)`.
   Q1 and the actor stay fixed; Q2 receives the shared-residual gradient through
   `Z_next` and the current-agreement gradient through `Z_cur`.
2. Recompute Q2's next values after its step, using the same rows and detached next
   actions. Update Q1 using `0.5 * mean(D**2) + alpha * 0.5 * mean((Y_next - sg(Z_next))**2)`,
   with Q2's bootstrap detached.
3. When the actor clock is due, minimize `-mean(Q2(s_next, actor(s_next)))`.
   Both critics' parameters stay fixed; gradients pass through Q2's action input
   to fresh actor actions.

Q2's own TD residual (`D_prime`) is diagnostic-only in this variant. It does not
replace the shared residual in either critic loss. This differs from the core's
opt-in `separate_td` objective.

| Setting | Packaged conditioned-paired value |
| --- | --- |
| Next-input agreement `alpha` | 0.01 |
| Current-input agreement `alpha_prime` | 10.0 |
| Critic-only preparation | First 5,000 learner iterations |
| Actor interval | Every 10 iterations after preparation, configurable to an integer >= 2 |
| Update order | Q2, fresh Q2 values for Q1, Q1, then actor if due |
| Actor input | Next states |
| Critic initialization | Independent |
| Targets / beta | Disabled |

Learner iterations are counted after replay warm-up, not from environment step
zero. Both critics update on every learner iteration. The actor's first default
update is at learner iteration 5,010. Termination and truncation remain separate;
the screening manifest bootstraps across truncations but not true terminations.

See the [implementation](../../src/three_network_ddpg/algorithms/paired_shared_d.py),
[configuration](../../src/three_network_ddpg/conf/algorithm/three_network_shared_d_conditioned_paired.yaml),
and [gradient, pairing, clock, and checkpoint tests](../../tests/unit/test_paired_shared_d.py).

## Inspect and plan

From the repository root, after creating the environment described in the
[README](../../README.md):

```bash
conda activate three-network-ddpg
python -m three_network_ddpg algorithms
python -m three_network_ddpg config validate --algorithm three_network_shared_d_conditioned_paired --environment pendulum_v1 --experiment three_network_core
python -m three_network_ddpg experiment validate experiments/manifests/pendulum_shared_d_conditioned_paired_20k_01.yaml
python -m three_network_ddpg experiment plan experiments/manifests/pendulum_shared_d_conditioned_paired_20k_01.yaml
```

These commands inspect or validate without training. The
[20k screening manifest](../../experiments/manifests/pendulum_shared_d_conditioned_paired_20k_01.yaml)
plans four runs: DDPG plus conditioned-paired actor intervals 2, 5, and 10.
Each run uses 20,000 environment steps and training seed 11, with five fixed
evaluation seeds. The total planned training budget is 80,000 environment steps.
This is a development screen, not a confirmatory study.

To execute that screen deliberately, use:

```bash
python -m three_network_ddpg experiment run experiments/manifests/pendulum_shared_d_conditioned_paired_20k_01.yaml
```

The direct `train-three-network` command selects the original core, not this
variant. Its short default budget is also shorter than the latest variant's
critic-only preparation. Use the manifest for this comparison.
Manifest `resume` starts a new attempt from scratch; loading an algorithm
checkpoint through the Python API is a separate operation.

## Evidence

Existing reports describe [the original pilot](pendulum_pilot_01_failure_analysis.md),
[separate-TD screening](separate_td_screening_analysis.md), and
[shared-D instability](shared_d_instability_correction.md). Their findings apply
to their recorded configurations. The latest conditioned-paired manifest and
synthetic tests do not themselves demonstrate improved Pendulum control.
Raw training artifacts are not committed.
