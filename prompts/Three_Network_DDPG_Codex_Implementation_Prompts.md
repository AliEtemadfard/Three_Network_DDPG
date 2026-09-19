# Codex Implementation Guide for the Three Network DDPG Research Project

## Purpose

This guide turns the current research specification into a staged Codex workflow. The first stage creates only the research repository, collaboration rules, configuration system, documentation, tests, and extension points. Later stages implement and verify the algorithms one at a time.

The central method is a continuous-control actor-critic system with one deterministic actor and two independently optimized critics. Its research objective is to investigate whether a shared Bellman residual can be reduced while policy improvement and critic agreement are preserved. The implementation must not claim global convergence or optimal-policy convergence. The current defensible target is a local or minibatch descent condition plus empirical stability evidence.

## Recommended Execution Strategy

Use the prompts in order. Do not ask Codex to build every algorithm in one pass. Each stage ends with tests, documentation updates, a diff review, and one local Git commit. Do not push to a remote until the team reviews the commit.

Before starting:

1. Create an empty project directory named `three-network-ddpg`.
2. Put the research specification in the project directory as `Three_Network_DDPG_Research_Specification.docx`, or attach it to the Codex task.
3. Open that directory in Codex, or run Codex CLI from it.
4. If using Codex CLI, run `/status` to verify the working directory and permissions. `/init` is optional because Prompt 1 explicitly creates a project-specific `AGENTS.md`.
5. Paste Prompt 1. Run later prompts only after reviewing the previous commit.

Example PowerShell startup:

```powershell
mkdir three-network-ddpg
cd three-network-ddpg
codex
```

Example Bash startup:

```bash
mkdir -p three-network-ddpg
cd three-network-ddpg
codex
```

## Research Constraints Codex Must Preserve

The implementation must use the following notation and gradient boundaries consistently:

- Actor: `mu_theta`.
- Critic One: `Q_phi`.
- Critic Two: `Q_phi_prime`.
- `Y = Q_phi(s, a)`.
- `Y1 = Q_phi(s_next, mu_theta(s_next))`.
- `Z1 = Q_phi_prime(s, a)`.
- `Z = Q_phi_prime(s_next, mu_theta(s_next))`.
- Shared residual: `delta = r + gamma * bootstrap_mask * Z - Y`.
- Shared squared error: `0.5 * mean(delta ** 2)`.
- Current-input agreement: `0.5 * mean((Z1 - Y) ** 2)`.
- Next-input agreement: `0.5 * mean((Y1 - Z) ** 2)`.
- Actor objective: maximize Critic Two's value, initially on next states, with current states available as an ablation.

During Critic Two's update, gradients may update only `phi_prime`; actor outputs and Critic One outputs are fixed targets. During Critic One's update, gradients may update only `phi`; actor outputs and Critic Two outputs are fixed targets. During the actor update, both critics' parameters are frozen, but the graph from Critic Two's action input back to the actor must remain connected.

The proposed main method has no target actor and no target critic. Target networks may exist only in standard DDPG, TD3, or an explicitly named ablation. Terminal and truncation handling must follow the environment API deliberately; time-limit truncation must not be silently treated as a true terminal state.

The following choices remain experimental and must be configurable rather than hard-coded:

- simultaneous, alternating, and guarded update schedules;
- no agreement, current-only, next-only, and both agreement penalties;
- no beta regularizer, original linear `Y`, squared `Y`, and squared `Y` plus `Z` regularizers;
- current-state or next-state actor objective;
- independent or identical critic initialization;
- actor update frequency;
- actor and critic learning rates;
- agreement coefficients;
- gradient clipping, disabled by default;
- acceptance batch and independent validation batch behavior for guarded updates.

## Target Repository Structure

```text
three-network-ddpg/
├── AGENTS.md
├── README.md
├── CONTRIBUTING.md
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md
├── CHANGELOG.md
├── environment.yml
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── config.yaml
│   ├── algorithm/
│   │   ├── ddpg.yaml
│   │   ├── three_network.yaml
│   │   ├── td3.yaml
│   │   ├── sac.yaml
│   │   ├── a2c.yaml
│   │   └── a3c.yaml
│   ├── environment/
│   │   ├── toy_continuous.yaml
│   │   └── pendulum_v1.yaml
│   └── experiment/
│       ├── smoke.yaml
│       ├── ddpg_baseline.yaml
│       └── three_network_ablation.yaml
├── docs/
│   ├── research/
│   │   ├── source/
│   │   │   └── Three_Network_DDPG_Research_Specification.docx
│   │   ├── problem_statement.md
│   │   ├── mathematical_notation.md
│   │   └── open_questions.md
│   ├── architecture.md
│   ├── experiment_protocol.md
│   ├── baseline_policy.md
│   ├── reproducibility.md
│   └── decisions/
│       └── 0001-repository-architecture.md
├── prompts/
│   └── README.md
├── src/
│   └── three_network_ddpg/
│       ├── __init__.py
│       ├── cli.py
│       ├── core/
│       │   ├── interfaces.py
│       │   ├── types.py
│       │   └── registry.py
│       ├── algorithms/
│       │   ├── ddpg/
│       │   ├── three_network/
│       │   ├── td3/
│       │   ├── sac/
│       │   ├── a2c/
│       │   └── a3c/
│       ├── networks/
│       ├── buffers/
│       ├── environments/
│       ├── training/
│       ├── evaluation/
│       ├── tracking/
│       ├── analysis/
│       └── utils/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── sweep.py
│   └── summarize_runs.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── notebooks/
│   └── README.md
├── data/
│   └── README.md
├── runs/
│   └── README.md
└── reports/
    ├── figures/
    └── tables/
```

Generated checkpoints, raw trajectories, TensorBoard event files, temporary environment files, and full run directories must be ignored by Git. Small aggregate tables and final figures may be committed deliberately. Do not add a license file until the team chooses a license.

## Prompt 1 Repository Bootstrap and Research Documentation

Paste the following prompt into Codex first.

```text
Role

Act as the lead research software engineer bootstrapping a collaborative deep reinforcement learning repository. Work conservatively: preserve supplied research materials, avoid destructive commands, and do not push to any remote.

Goal

Create a clean, extensible, Git-based Python research repository for the supplied Three Network DDPG specification. This stage is infrastructure and documentation only. Do not implement full DDPG, TD3, SAC, A2C, A3C, or the proposed algorithm. Create interfaces, registries, configuration placeholders, CLI help, and small validation tests only where needed to prove the scaffold is coherent.

Required first actions

1. Inspect the current directory and Git status before changing anything.
2. Locate and read `Three_Network_DDPG_Research_Specification.docx`. If it is unavailable, stop and ask for its path rather than inventing the method.
3. Extract its text with an available local tool such as Pandoc or Python. Preserve the original DOCX by placing a copy under `docs/research/source/`.
4. If this is not already a Git repository, initialize it with `main` as the default branch. Create and switch to `chore/bootstrap-research-repo`. If Git already exists, preserve its history and current changes; do not overwrite unrelated work.
5. State a concise plan before editing.

Research interpretation to preserve

- The method uses one deterministic actor and two independently optimized critics.
- Critic One represents the current side of a shared Bellman residual; Critic Two supplies the bootstrap value and actor objective.
- The proposed main variant has no target actor and no target critic.
- Both critic-agreement penalties, the beta regularizer, and the update schedule are unresolved experimental choices.
- The actor step can increase the shared error, and consistency penalties can rotate critic updates away from the shared-error descent direction.
- The project must not claim global convergence or convergence to an optimal policy.
- The first credible target is correct gradient flow, local or minibatch descent diagnostics, and reproducible empirical comparison.

Repository requirements

Create the repository structure described below, adapting names only when there is a concrete technical reason. Use a `src/` package layout with package name `three_network_ddpg`.

Create:

- `AGENTS.md` with project-specific instructions for future Codex sessions.
- A comprehensive `README.md`.
- `CONTRIBUTING.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, and `CHANGELOG.md`.
- `environment.yml` for a Conda environment named `three-network-ddpg`.
- `pyproject.toml` with package metadata and separate core, development, plotting, and optional experiment-tracking dependencies.
- `.gitignore`, `.pre-commit-config.yaml`, and a lightweight CPU CI workflow.
- composable YAML configuration groups for algorithm, environment, and experiment settings. Prefer Hydra/OmegaConf if it stays simple and testable.
- documentation pages for the problem statement, notation, architecture, experiment protocol, reproducibility rules, baseline policy, open questions, and the initial architecture decision record.
- minimal package modules for core interfaces, typed transition batches, an algorithm registry, CLI help, config validation, and output-directory metadata.
- algorithm directories for DDPG, the three-network method, TD3, SAC, A2C, and A3C, but label unimplemented algorithms explicitly. Do not create fake training logic.
- test directories and a few fast tests that verify imports, configuration composition, registry metadata, CLI help, and seed utility behavior.
- `data`, `runs`, `notebooks`, `reports`, and `prompts` guidance files explaining what belongs there.

README requirements

The README must be useful to a new teammate without a verbal explanation. Include:

1. Project title, current status, and a two-paragraph problem statement.
2. The research hypothesis and explicit non-claims.
3. A small architecture diagram using Mermaid.
4. Definitions of actor, both critics, Y, Y1, Z1, Z, the shared residual, and both agreement errors.
5. A clear warning that the beta term and convergence statement are provisional.
6. A roadmap with phases: verified DDPG; minimal shared-error method; agreement ablations; beta and schedule ablations; multi-seed studies; theory refinement; novelty review.
7. Conda setup commands, environment activation, editable installation, tests, linting, and CLI help.
8. Commands that currently work. Mark future training commands as planned; do not pretend unfinished algorithms run.
9. Repository layout and extension instructions for adding a new algorithm or environment.
10. Experiment configuration and output conventions.
11. Reproducibility rules: seeds, environment versions, Git commit hash, full resolved configuration, library versions, hardware metadata, evaluation without exploration noise, and raw-versus-aggregate result policy.
12. Team Git workflow with feature branches, pull requests, review, conventional commits, and no generated artifacts in Git.
13. Baseline policy explaining that DDPG and TD3 are primary direct comparisons. A2C/A3C are later on-policy contextual baselines and are not architecturally equivalent to DDPG. SAC is a useful continuous-control reference but changes the policy and entropy assumptions.
14. Open research questions copied faithfully in paraphrased form from the specification.

AGENTS.md requirements

Tell future coding agents to:

- read the research problem and notation documents before algorithm edits;
- preserve the exact stop-gradient matrix;
- keep proposed-method target networks disabled unless an ablation explicitly enables them;
- handle terminated and truncated transitions deliberately;
- avoid hidden defaults and magic numbers;
- make every experimental choice configurable;
- keep training, evaluation, environment wrappers, algorithms, tracking, and plotting separated;
- add focused tests for every gradient-path or update-order change;
- update relevant documentation and changelog entries;
- never claim theoretical results not established by the project;
- never copy external baseline code without recording source URL, commit or version, license, and local modifications in `THIRD_PARTY_NOTICES.md`;
- inspect Git diff and run relevant checks before committing;
- never push, force-push, rewrite history, delete user work, or change remotes without explicit approval.

Technical conventions

- Target Python 3.11 unless local compatibility requires another supported version and the change is documented.
- Use PyTorch and Gymnasium for later implementations.
- Make CPU execution the default. Keep CUDA optional and do not hard-code device assumptions.
- Use type hints for public interfaces and dataclasses or typed mappings for batches and metrics.
- Prefer CSV or JSONL plus TensorBoard for default local tracking; make Weights and Biases optional.
- Store a resolved config, seed, Git commit, package versions, environment ID, timestamps, and hardware metadata with every run.
- Use deterministic seeding where supported, while documenting that full determinism is not guaranteed across every device and operation.
- Keep test and lint commands fast enough for ordinary pull requests.

Configuration design

The configuration schema must reserve fields for:

- algorithm name and implementation status;
- environment ID and wrappers;
- total steps, warm-up steps, batch size, replay capacity, evaluation frequency, and evaluation episodes;
- gamma, actor learning rate, critic learning rates, optimizer settings, hidden layers, activations, device, and seed;
- exploration noise and action rescaling;
- target-network settings for baseline algorithms only;
- agreement mode: none, current, next, or both;
- alpha and alpha-prime;
- beta regularizer mode: none, linear_y, l2_y, or l2_y_z;
- update schedule: simultaneous, alternating, or guarded;
- actor state source: current or next;
- critic initialization: independent or identical;
- actor update delay;
- guarded-update tolerance, backtracking factor, maximum retries, minimum learning rate, and validation-batch option;
- diagnostic frequency for gradient alignment metrics.

Architecture contract

Define a small algorithm interface that can later support both off-policy and on-policy methods without forcing them into the same internal training logic. It should cover action selection, observation of transitions or rollout data, update execution, evaluation mode, state serialization, checkpoint restoration, and structured metrics. Use capability metadata such as `is_on_policy`, `uses_replay_buffer`, `supports_continuous_actions`, and `implementation_status`.

Baseline policy

Do not vendor external baseline implementations in this stage. Document two allowed later paths:

1. a clean, tested implementation under the common project interface; or
2. a thin adapter to a maintained library for verification.

Any copied or adapted code requires license and provenance review. DDPG must be implemented and verified first. TD3 is the next direct strong baseline. SAC, A2C, and A3C come later. A3C requires a separate multiprocessing design and must not be treated as a trivial variant of the off-policy trainer.

Validation and completion bar

Before finishing:

1. Run formatting or lint checks that are available without a large dependency installation.
2. Run the fast tests if the local environment supports them. If dependencies are not installed, run syntax and file-structure checks and report exactly what remains unverified.
3. Inspect `git status` and `git diff --check`.
4. Confirm that no generated outputs, caches, secrets, large binaries other than the supplied research specification, or environment directories are staged.
5. Review README commands for truthfulness.
6. Create one local commit named `chore: bootstrap reproducible RL research repository` only if all staged changes are intentional and validation has passed. Do not push.

Final response

Summarize the files created, the checks run, the resulting commit hash, any assumptions, and the exact next recommended prompt. Mention unresolved questions rather than silently choosing research conclusions.
```

## Prompt 2 Core Interfaces and Test Harness

Run this only after Prompt 1 is reviewed.

```text
Implement the smallest executable research core for this repository without implementing a learning algorithm yet.

First read `AGENTS.md`, `README.md`, the research documentation, and the current Git status. Create a feature branch named `feat/core-training-contracts` from the reviewed bootstrap commit.

Required outcomes:

- Finalize typed contracts for transition batches, rollout batches, algorithm capabilities, actions, update results, and metric records.
- Implement deterministic seed utilities for Python, NumPy, PyTorch, Gymnasium action spaces, and environment reset.
- Implement continuous-action scaling and clipping utilities with inverse-scaling tests.
- Implement a replay buffer with correct shapes, dtypes, device transfer, terminated flags, truncated flags, and bootstrap-mask policy.
- Implement environment creation for a tiny deterministic continuous-control test environment and `Pendulum-v1`.
- Implement run-directory creation and metadata capture, including resolved config and Git commit.
- Implement a trainer shell that delegates behavior to an algorithm interface and does not contain algorithm-specific loss logic.
- Implement a separate evaluator with exploration disabled.
- Add unit and smoke tests, including a very short random-policy environment interaction.
- Keep A3C and other on-policy algorithms out of the off-policy replay loop through capability-aware interfaces.
- Update README and architecture documentation to match the real behavior.

Do not implement DDPG or the proposed losses. Do not add target networks. Do not silently convert truncation into termination. Run relevant tests and linting, inspect the diff, and commit locally as `feat: add core training contracts and test harness`. Do not push.
```

## Prompt 3 Verified Standard DDPG Baseline

```text
Implement and verify the standard DDPG baseline as the first real algorithm.

Read `AGENTS.md`, the baseline policy, experiment protocol, and current tests. Work on branch `feat/ddpg-baseline`. Preserve unrelated changes.

Requirements:

- Implement a deterministic actor, online critic, target actor, and target critic.
- Implement replay-based critic TD regression, deterministic policy gradient, exploration noise for collection only, evaluation without noise, and soft target updates.
- Keep network construction configurable and reuse shared action-scaling utilities.
- Add tests for output shapes, action bounds, terminal masking, truncation policy, soft updates, actor gradient flow, critic isolation, checkpoint round trips, deterministic evaluation, and a short Pendulum smoke run.
- Log episodic return, episode length, critic loss, actor loss, Q statistics, gradient norms, environment steps, update steps, wall time, and evaluation metrics.
- Add a tiny overfit or deterministic-transition test that catches reversed loss signs and broken targets.
- Add `configs/experiment/ddpg_baseline.yaml` with conservative smoke defaults, not paper-scale defaults.
- Ensure the baseline is faithful before optimizing performance.
- Update README commands and record implementation status.

Run tests, lint, and a short CPU smoke experiment. Save the resolved config and metadata. Do not present a single smoke run as scientific evidence. Commit locally as `feat: implement verified DDPG baseline`. Do not push.
```

## Prompt 4 Minimal Three Network Shared Error Variant

```text
Implement the minimal proposed three-network algorithm with the shared TD error only. Do not add agreement penalties, beta regularization, or guarded updates yet.

Work on branch `feat/three-network-core`. Read all research and architecture documentation before editing.

Definitions:

- `Y = Q_phi(s, a)`
- `Y1 = Q_phi(s_next, mu_theta(s_next))`
- `Z1 = Q_phi_prime(s, a)`
- `Z = Q_phi_prime(s_next, mu_theta(s_next))`
- `delta = r + gamma * bootstrap_mask * Z - Y`
- shared loss is `0.5 * mean(delta ** 2)`
- actor loss is initially `-mean(Q_phi_prime(s_next, mu_theta(s_next)))`

Requirements:

- One actor and two independently initialized critics, with independent optimizers.
- No target actor and no target critic in this variant.
- Implement sequential and simultaneous schedules. In simultaneous mode, compute all gradients from one pre-update snapshot and apply optimizer steps only after all required gradients have been stored.
- During critic updates, actor actions are detached. Each critic treats the other critic's relevant outputs as fixed targets.
- During actor updates, freeze critic parameters without detaching the critic output from the actor action.
- Add explicit tests proving which parameter groups receive gradients in each phase.
- Add tests that fail if target-network modules appear in the proposed variant.
- Log signed residual mean, absolute residual mean, shared squared error, Y/Y1/Z1/Z statistics, Q magnitude, gradient norms, and return.
- Add optional diagnostics for the actor/shared-error gradient interaction without changing the training gradient.
- Add a current-state versus next-state actor-input configuration field, but keep next-state as the reconstructed default.
- Add checkpointing and configuration round-trip tests.
- Run a tiny deterministic-environment test and a short Pendulum smoke run.

Document every deviation from the research specification. Do not claim convergence. Commit locally as `feat: implement minimal three-network shared-error method`. Do not push.
```

## Prompt 5 Agreement Penalties Beta Variants and Update Schedules

```text
Extend the three-network method with controlled ablations. Work on branch `feat/three-network-ablations`.

Implement configurable agreement modes:

- none;
- current only: compare `Z1` with detached `Y` during Critic Two's update;
- next only: compare `Y1` with detached `Z` during Critic One's update;
- both.

Implement beta regularizer modes:

- none;
- the original linear `Y` proposal, clearly labeled experimental;
- squared `Y` magnitude;
- squared `Y` and `Z` magnitude.

Implement update schedules:

- simultaneous;
- alternating or sequential with intentional recomputation after earlier steps;
- guarded updates with snapshot, proposal, evaluation, backtracking, acceptance or restoration.

For guarded updates, make it configurable whether guarding applies to the whole joint update, the critic phase, or the actor phase. Support an independent validation batch for monitoring. Record accepted steps, rejected steps, retry count, effective learning rate, error before and after, and validation-batch change. Keep optimizer-state restoration correct when a proposal is rejected.

Add focused tests for detach behavior, schedule semantics, same-snapshot simultaneous gradients, sequential recomputation, restoration after rejection, and beta formulas. Add diagnostics for critic-objective alignment with the shared-error gradient and actor interaction. Expensive gradient diagnostics must run at a configurable interval.

Update the experiment protocol and configuration examples. Run unit tests and smoke tests only; do not launch a large sweep. Commit locally as `feat: add three-network ablations and guarded updates`. Do not push.
```

## Prompt 6 Experiment Runner and Statistical Protocol

```text
Build the reproducible experiment and analysis layer on branch `feat/experiment-protocol`.

Requirements:

- Support multi-seed runs without embedding seeds in source code.
- Generate a manifest for every run containing algorithm, environment, seed, resolved configuration, Git commit, environment and package versions, hardware, start/end times, and completion status.
- Store append-safe JSONL or CSV metrics and TensorBoard events.
- Produce aggregate tables with mean, standard deviation, median, interquartile range, confidence intervals, success rate, divergence count, wall time, and peak memory when available.
- Plot individual seeds faintly and aggregate curves clearly. Never show only the best seed.
- Define divergence and failed-run handling before aggregation.
- Compare learning curves on common environment-step axes.
- Implement the initial matrix: standard DDPG; DDPG without target actor; shared-error only; current agreement only; next agreement only; both agreements; guarded variant; TD3 when ready.
- Keep A2C/A3C/SAC outside the primary direct-comparison table until their protocols are implemented and budgets are made comparable.
- Provide dry-run and smoke modes that validate commands without expensive experiments.
- Add tests for manifest creation, incomplete-run recovery, aggregation, confidence intervals, and plotting from synthetic data.
- Update README with exact commands, expected outputs, and the distinction between smoke checks and scientific experiments.

Do not start a long experiment sweep automatically. Commit locally as `feat: add reproducible experiment and analysis pipeline`. Do not push.
```

## Prompt 7 Additional Baselines

```text
Add additional baselines one at a time, not in a single uncontrolled change. Start with TD3, then SAC, then A2C, and finally A3C. Use one feature branch and one reviewed commit per algorithm.

Before implementing each baseline:

1. Record whether it is a direct architectural comparison or a contextual performance baseline.
2. Decide between a clean in-repository implementation and a maintained-library adapter.
3. If code is copied or adapted, record its exact source, version or commit, license, and modifications in `THIRD_PARTY_NOTICES.md` before merging.
4. Match environment wrappers, action scaling, evaluation episodes, total environment steps, seeds, and logging conventions where scientifically appropriate.
5. Add algorithm-specific correctness tests and a CPU smoke run.

TD3 is the strongest next direct baseline because it is an off-policy deterministic continuous-control algorithm with twin critics. SAC is a useful stochastic off-policy comparison but changes the entropy objective. A2C and A3C are on-policy methods and must use rollout storage rather than the replay-buffer trainer. A3C additionally requires asynchronous multiprocessing and should have tests for worker seeding, shared-state synchronization, shutdown, and exception propagation.

Do not interpret better or worse returns from A2C/A3C as evidence about the shared-error mechanism without accounting for the on-policy/off-policy difference. Update documentation and commit each baseline separately. Do not push.
```

## Suggested Team Workflow

Use short-lived branches and pull requests:

```text
main
├── chore/bootstrap-research-repo
├── feat/core-training-contracts
├── feat/ddpg-baseline
├── feat/three-network-core
├── feat/three-network-ablations
├── feat/experiment-protocol
└── feat/baseline-td3
```

Each pull request should contain one scientific or infrastructure change, its tests, configuration changes, and documentation. Avoid mixing refactors with algorithmic changes because doing so makes regression analysis difficult.

Recommended issue labels:

- `algorithm`
- `experiment`
- `theory`
- `baseline`
- `reproducibility`
- `bug`
- `documentation`
- `needs-advisor-decision`

## Initial Experiment Priority

1. Verify standard DDPG on the toy continuous environment and Pendulum.
2. Verify gradient paths in the shared-error-only method.
3. Compare simultaneous and sequential schedules on debugging metrics, not only return.
4. Add current agreement and next agreement separately.
5. Compare beta regularizers.
6. Add guarded actor-only updates before guarding the whole joint step.
7. Run at least ten seeds only after smoke tests and short stability runs pass.
8. Add TD3.
9. Add SAC and on-policy methods later for context.

This order isolates the source of failures. If everything is implemented at once, it becomes difficult to determine whether instability comes from a loss formula, detach mistake, update ordering, environment handling, or experiment configuration.

## Review Checklist After Every Codex Stage

- Does `git diff --check` pass?
- Are all modified files within the requested scope?
- Do README commands actually work?
- Are unfinished features labeled unfinished?
- Are `terminated` and `truncated` handled explicitly?
- Are action ranges and dtypes correct?
- Do tests verify gradient presence and absence, not only output shapes?
- Are proposed-method target networks absent unless the config names an ablation?
- Are random seeds, resolved config, Git commit, versions, and environment IDs saved?
- Are generated outputs excluded from Git?
- Are third-party sources and licenses documented?
- Does the commit contain one coherent change?
- Are theoretical statements limited to what has actually been established?

## Final Recommendation

Use Prompt 1 now. Its successful output should be a documented, testable repository scaffold with a truthful README and no pretend implementations. Review that commit with teammates before Prompt 2. This creates a stable shared vocabulary and prevents later algorithm code from locking the project into an unsuitable training loop.
