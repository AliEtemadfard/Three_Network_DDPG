# Three Network DDPG

**Status:** Research repository scaffold. The CLI inspects algorithm metadata and validates configuration; no training algorithm is implemented. The [original research specification](docs/research/source/Three_Network_DDPG_Research_Specification.docx) is preserved verbatim.

DDPG uses a deterministic actor and a value critic, usually with slowly updated target copies of both networks. Target networks reduce movement in the bootstrap target but do not establish general convergence for nonlinear actor–critic learning. They also add parameter storage. This project asks whether one actor and two independently optimized critics can maintain a useful Bellman relation without target networks in the proposed main variant.

The proposal gives Critic One the current side of a shared temporal difference residual and Critic Two the bootstrap side and actor objective. Matching the critics on current and next inputs may prevent incompatible value estimates, yet the actor update can increase the residual and either agreement penalty can rotate a critic update away from residual descent. We will test gradient flow, local or minibatch descent diagnostics, and reproducible control performance before attempting a theory claim.

## Hypothesis and limits

**Research hypothesis:** A target-free, three-network deterministic actor–critic method may maintain useful critic agreement and improve empirical stability or cost relative to verified DDPG under an appropriate update schedule and regularizer. This is a question to test, not an established result.

**Non-claims:** We do not claim global convergence, convergence to an optimal policy, monotonic return improvement, lower compute cost, or novelty. A descent result on one acceptance minibatch does not imply descent on the replay distribution or better return. The beta regularizer and any convergence statement are **provisional**.

## Architecture and notation

```mermaid
flowchart LR
    B["Replay batch: s, a, r, s', terminated, truncated"] --> C1["Critic One Qφ"]
    B --> C2["Critic Two Qφ′"]
    B --> A["Actor μθ"]
    A --> N["Next action μθ(s')"]
    N --> C1
    N --> C2
    C1 --> Y["Y current; Y1 next"]
    C2 --> Z["Z1 current; Z next"]
    Y --> E["Shared residual and agreement diagnostics"]
    Z --> E
    Z --> O["Actor objective"]
```

For a batch of transitions `(s, a, r, s', terminated, truncated)`, `μθ` is the deterministic actor. `Qφ` is Critic One and `Qφ′` is Critic Two; each critic has its own optimizer. Let `a' = μθ(s')`. `Y = Qφ(s,a)` and `Z1 = Qφ′(s,a)` use identical current inputs. `Y1 = Qφ(s',a')` and `Z = Qφ′(s',a')` use identical next inputs.

With an explicitly chosen bootstrap mask `m`, the shared residual is `δ = r + γ m Z − Y` and its diagnostic error is `e = ½ mean(δ²)`. Current agreement error is `½ mean((Z1 − Y)²)`; next agreement error is `½ mean((Y1 − Z)²)`. `m` must handle termination and truncation deliberately; see [notation](docs/research/notation.md). The board derivation omitted terminal masking, so implementation must add it. The actor maximizes Critic Two on next states in the initial interpretation, with current states reserved as an ablation.

**Warning:** The original linear beta term can drive negative `Y` values downward and does not bound their magnitude. Linear and quadratic variants are research alternatives. The final objective and convergence statement remain unresolved; see [open questions](docs/research/open_questions.md).

## Roadmap

1. Implement and verify standard DDPG on a small continuous-control task, then Pendulum.
2. Implement the minimal three-network shared-error method and verify every gradient path.
3. Ablate no, current, next, and both critic-agreement penalties.
4. Ablate beta regularizers, simultaneous/alternating/guarded schedules, actor state source, initialization, and actor delay.
5. Run fair multi-seed studies and failure analysis.
6. Refine a local descent theory for the selected implementation and its actual assumptions.
7. Review novelty against target-free, residual-gradient, and twin-critic literature before framing a contribution.

## Set up and commands that work now

Use Python 3.11 and CPU by default. The environment file installs the editable package with development tools. PyTorch and Gymnasium are declared for later implementations; this scaffold does not create an environment or train an agent.

~~~bash
conda env create -f environment.yml
conda activate three-network-ddpg
python -m pip install -e ".[dev]"
three-network-ddpg --help
three-network-ddpg algorithms
three-network-ddpg config validate
three-network-ddpg config show
python -m pytest
ruff check .
ruff format --check .
~~~

For local TensorBoard or plotting support, install `.[tracking]` or `.[plotting]`. Weights & Biases remains optional under `.[wandb]`. No account or network service is required for default local tracking.

**Planned, currently unavailable:** training, evaluation, replay collection, checkpointing, and plotting commands. The registry reports every algorithm as `planned`; it will never imply that a placeholder is trainable.

## Layout and extension points

| Path | Purpose |
| --- | --- |
| `src/three_network_ddpg/core/` | Typed batches, algorithm interface, seed and metadata utilities |
| `src/three_network_ddpg/algorithms/` | Metadata registry and explicit placeholders for DDPG, three-network, TD3, SAC, A2C, A3C |
| `src/three_network_ddpg/conf/` | Composable algorithm, environment, experiment YAML |
| `docs/research/` | Problem, notation, architecture, protocol, reproducibility, baseline policy, questions, decisions |
| `tests/` | Fast scaffold checks; later gradient and integration tests |
| `data/`, `runs/`, `reports/` | Local inputs and generated outputs, ignored except guidance files |
| `notebooks/`, `prompts/` | Exploration and reusable research prompts |

To add an algorithm, first document its capability metadata and experimental role, add a configuration group, then implement the common interface without forcing on-policy rollouts through replay-buffer logic. Register a factory only after behavior and serialization tests pass. To add an environment, add an environment YAML group with ID, wrapper order, observation and reward treatment, action bounds policy, and a small environment smoke test. Keep training, evaluation, wrappers, tracking, and plotting separate.

## Configurations and outputs

`conf/config.yaml` selects one algorithm, environment, and experiment group. `config validate` composes them with OmegaConf and checks the required schema and cross-field constraints. The default proposed-method config is a **design placeholder**, not a selected scientific conclusion. It reserves alpha, alpha-prime, beta mode, schedule, state source, initialization, guarded-update controls, actor delay, and diagnostic frequency. DDPG and TD3 configs reserve their own target-network settings; the proposed main config has none. Device is `cpu` unless explicitly changed.

When training exists, write each run under `runs/<experiment>/<algorithm>/<environment>/<timestamp>-seed-<seed>/`. Save the fully resolved config, seed, Git commit, package and environment versions, UTC start/end timestamps, CPU/GPU and OS metadata, raw per-step or per-episode CSV/JSONL, optional TensorBoard events, and checkpoints. Keep aggregates and figures under `reports/`, linked to raw run IDs. The metadata utility already creates the common JSON record; it does not start a run.

## Reproducibility and comparison policy

Seed Python, NumPy, PyTorch, action spaces, and environments where supported. Record the actual seeds and deterministic backend settings; exact determinism is not guaranteed across devices or operations. Evaluation uses a separate seeded environment and **no exploration noise**. Record wrapper order, Gymnasium version and environment ID, Git commit hash, resolved configuration, package versions, hardware, timestamps, and failure or rejection counts. Retain every seed's raw results; aggregate only after all planned seeds, and report uncertainty and divergence rather than choosing successful runs.

DDPG and TD3 are the primary direct continuous-control comparisons. A2C and A3C are later on-policy contextual baselines, architecturally different from DDPG; A3C needs a separate multiprocessing design. SAC is a useful continuous-control reference but changes policy stochasticity and entropy assumptions. No external baseline code is vendored. Later paths are a clean tested implementation under the common interface or a thin adapter to a maintained library; copied or adapted code requires provenance and license review. See [baseline policy](docs/research/baselines.md).

## Team workflow

Branch from `main` for each feature, open a pull request, request review, and use conventional commits. Keep changes focused and run fast tests and lint before review. Never commit generated datasets, runs, checkpoints, environments, caches, or secrets. Document configuration and evidence for research choices; update the changelog. Do not push or rewrite Git history without explicit authorization.

## Open research questions

The source asks whether removing the target actor preserves stability and return; whether independent critics can replace the online/target critic relationship; which agreement constraint is needed; when shared error is a valid local Lyapunov candidate; whether acceptance/backtracking can enforce minibatch descent without freezing improvement; whether any parameter, memory, or time benefit is measurable; and whether the principle transfers to DQN after continuous-control validation. Advisor decisions remain open on current versus next actor states, any soft-update role for Critic Two, linear versus quadratic beta or weight regularization, simultaneous versus alternating updates, the scope of a theorem, whether both agreement terms are required, and whether a guard covers the whole update or actor step. See [open questions](docs/research/open_questions.md).

## Research documents

Start with [problem](docs/research/problem.md), [notation](docs/research/notation.md), and [architecture](docs/research/architecture.md). Then read the [experiment protocol](docs/research/experiments.md), [reproducibility rules](docs/research/reproducibility.md), [baseline policy](docs/research/baselines.md), and [initial decision record](docs/decisions/0001-research-scaffold.md).
