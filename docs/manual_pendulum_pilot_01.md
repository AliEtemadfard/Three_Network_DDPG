# Manual Pendulum Pilot 01

This guide prepares three manually launched `Pendulum-v1` development stages. It does not authorize an automatic launch, a confirmatory run, or an interpretation beyond pilot diagnostics. Read [the frozen pilot plan](../experiments/plans/pendulum_pilot_01.yaml) before starting.

All commands below use the installed CLI syntax verified for the canonical Conda environment. Run them in a foreground PowerShell window so `Ctrl+C` can be handled by the runner. Do not use `--restart` unless intentionally archiving an incompatible study after preserving its artifacts.

## Preflight

```powershell
conda activate three-network-ddpg
python --version
python -m pip show three-network-ddpg
git rev-parse HEAD
git status --short --branch
```

The intended preparation commit must be recorded in the final pilot artifacts. Start only with a clean working tree. The canonical environment is Python 3.11; CPU and one worker are frozen in every pilot manifest.

## Stage A: pipeline smoke

Stage A has three variants and one paired development seed: standard DDPG, DDPG without a target actor, and target-free three-network shared-error-only sequential. It plans 3 runs x 2,000 training steps = 6,000 training environment steps. Each run evaluates 5 fixed episodes every 500 steps (20 scheduled evaluation episodes per run, 60 total). Output: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-a`. Execution is sequential (`workers: 1`).

```powershell
$StageA = "experiments/manifests/pendulum_pilot_01_stage_a.yaml"
three-network-ddpg experiment validate $StageA
three-network-ddpg experiment plan $StageA
```

Inspect the printed plan before manually launching. Do not interpret this stage as a learning comparison.

```powershell
# Manually start Stage A only after the preceding inspection.
three-network-ddpg experiment run $StageA
three-network-ddpg experiment status $StageA
three-network-ddpg experiment aggregate $StageA
three-network-ddpg experiment plot $StageA --metric evaluation_return_mean
three-network-ddpg experiment plot $StageA --metric shared_squared_error
```

Stop and inspect every Stage A artifact and status before considering Stage B. If a foreground run must be stopped, press `Ctrl+C` once and wait for it to return. The attempt is marked `interrupted`; do not delete it. Resume only retryable states with:

```powershell
three-network-ddpg experiment resume $StageA
three-network-ddpg experiment status $StageA
```

`experiment run` is the run-only-missing command: it selects only never-started run IDs. `experiment resume` selects incomplete, interrupted, and failed IDs. Divergent and non-finite outcomes remain terminal evidence and are not silently retried.

## Stage B: mechanism pilot

Stage B has three paired seeds (`11`, `22`, `33`) for standard DDPG, DDPG without a target actor, and three-network shared-error-only sequential. It plans 9 runs x 20,000 training steps = 180,000 training environment steps. Each run evaluates 5 fixed episodes every 1,000 steps (100 scheduled evaluation episodes per run, 900 total). Output: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-b`. Execution is sequential (`workers: 1`).

```powershell
$StageB = "experiments/manifests/pendulum_pilot_01_stage_b.yaml"
three-network-ddpg experiment validate $StageB
three-network-ddpg experiment plan $StageB
```

Only after Stage A is complete and inspected, manually launch Stage B:

```powershell
three-network-ddpg experiment run $StageB
three-network-ddpg experiment status $StageB
three-network-ddpg experiment aggregate $StageB
three-network-ddpg experiment plot $StageB --metric evaluation_return_mean
three-network-ddpg experiment plot $StageB --metric shared_squared_error
three-network-ddpg experiment plot $StageB --metric q_absolute_mean
```

`normalized_auc` is a report summary rather than a time-series plot metric; inspect it in `reports/aggregate.json`. For an interruption, keep the attempt, inspect `status`, and run `three-network-ddpg experiment resume $StageB`. Stop and inspect all Stage B results before running Stage C.

## Stage C: agreement pilot

Stage C has four target-free three-network variants—`none`, `current`, `next`, and `both`—on the same paired seeds (`11`, `22`, `33`). It plans 12 runs x 20,000 training steps = 240,000 training environment steps. Each run evaluates 5 fixed episodes every 1,000 steps (100 scheduled evaluation episodes per run, 1,200 total). Output: `runs/pendulum-pilot-01/pendulum-pilot-01-stage-c`. Execution is sequential (`workers: 1`).

```powershell
$StageC = "experiments/manifests/pendulum_pilot_01_stage_c.yaml"
three-network-ddpg experiment validate $StageC
three-network-ddpg experiment plan $StageC
```

Only after Stage B has been inspected, manually launch Stage C:

```powershell
three-network-ddpg experiment run $StageC
three-network-ddpg experiment status $StageC
three-network-ddpg experiment aggregate $StageC
three-network-ddpg experiment plot $StageC --metric evaluation_return_mean
three-network-ddpg experiment plot $StageC --metric shared_squared_error
three-network-ddpg experiment plot $StageC --metric current_agreement_error
three-network-ddpg experiment plot $StageC --metric next_agreement_error
three-network-ddpg experiment plot $StageC --metric q_absolute_mean
three-network-ddpg experiment plot $StageC --metric actor_gradient_norm
three-network-ddpg experiment plot $StageC --metric critic_one_gradient_norm
three-network-ddpg experiment plot $StageC --metric critic_two_gradient_norm
```

All Stage C conditions are fixed except agreement mode and its documented initial coefficients: none `(0.0, 0.0)`, current `(0.0, 0.1)`, next `(0.1, 0.0)`, and both `(0.1, 0.1)` for `(alpha, alpha_prime)`. There is no alpha sweep, beta is disabled, the schedule is sequential, the actor objective uses next states, critics are independently initialized, and the validation batch is disabled. If validation-batch metrics are later enabled in another named guarded study, they are monitoring-only and cannot accept or reject an update.

## Artifact inspection and collection

For any stage, replace `$StageRoot` with its output directory:

```powershell
$StageRoot = "runs/pendulum-pilot-01/pendulum-pilot-01-stage-a"
Get-ChildItem $StageRoot -Recurse -File | Select-Object FullName
Get-Content "$StageRoot/manifest.lock.json"
Get-Content "$StageRoot/matrix.json"
Get-Content "$StageRoot/reports/aggregate.json"
```

Collect these exact artifacts for later analysis: `manifest.lock.json`, `manifest.source.json`, `matrix.json`, every run's `run_spec.json`, `resolved_config.yaml`, `status.json`, every attempt's `STARTED.json`, `metadata.json`, `metrics.jsonl`, `summary.json`, `checkpoint.pt` when present, and terminal `COMPLETED.json`, `FAILED.json`, `DIVERGENT.json`, `NON_FINITE.json`, or `INTERRUPTED.json` markers. Also collect `reports/aggregate.json`, `reports/plots/index.json`, and all generated SVG figures.

The current runner writes append-safe `metrics.jsonl`; it does **not** produce CSV files or TensorBoard events with these manifests (`tracking.csv_jsonl=true`, `tensorboard=false`). Do not invent or expect CSV files. Failed, divergent, non-finite, interrupted, and incomplete statuses are part of the report and must be retained.

## Runtime estimate before Stage B or C

After Stage A, read each completed attempt's `summary.json` and `metrics.jsonl` for `wall_time_seconds` and `environment_steps_per_second`. Use the lowest observed Stage A throughput as a cautious baseline:

```text
estimated training time = planned training environment steps / conservative observed steps per second
```

Add a buffer for five-episode evaluations, setup, checkpointing, and interruptions. This is an estimate only; do not claim an exact runtime. Stage B and C must remain sequential and use their frozen budgets and seed counts.

## Interpretation boundary

Primary control metrics are evaluation return and normalized AUC. Mechanism metrics include shared residual/error, critic agreement, Q-value scale, and defined gradient-alignment diagnostics. Reliability metrics include non-finite failures, divergence, variance, and incomplete runs. A lower shared TD error alone does not imply a better policy. Do not select the best single seed, claim convergence or superiority, consume confirmatory seeds, or present this pilot as paper-level evidence. `l2_y_z` remains excluded.

Three-network mechanism metrics are undefined for the DDPG variants. A mechanism plot can therefore have no DDPG series; do not create proxy values or treat those diagnostics as directly comparable across algorithms. Compare all variants on the primary control and reliability metrics instead.
