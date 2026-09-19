# Notation and provisional objectives

For a replay minibatch `(s, a, r, s', terminated, truncated)`, the deterministic actor is `μθ`; Critic One is `Qφ`; Critic Two is `Qφ′`. Let `a' = μθ(s')`, detached during critic updates. The four vectors have batch length `B`:

| Vector | Definition | Role |
| --- | --- | --- |
| `Y` | `Qφ(s,a)` | Current side of shared residual |
| `Y1` | `Qφ(s',a')` | Next-input agreement |
| `Z1` | `Qφ′(s,a)` | Current-input agreement |
| `Z` | `Qφ′(s',a')` | Bootstrap and actor objective |

Use a documented mask `m` in `δ = r + γ m Z − Y` and diagnostic `e = ½ mean(δ²)`. A terminal transition has `m=0`. For truncations, `bootstrap_on_truncation` explicitly selects whether to bootstrap; the eventual implementation must verify that the final observation is available when bootstrapping. Never infer truncation from termination. The source board expression omitted this mask.

Current agreement is `c_current = ½ mean((Z1 − Y)²)`. Next agreement is `c_next = ½ mean((Y1 − Z)²)`. For Critic Two, `Y` is detached; its provisional loss is `e + alpha_prime * c_current`. For Critic One, `Z` is detached; its provisional loss is `e + alpha * c_next + beta_regularizer`. The actor maximizes `mean(Qφ′(s', μθ(s')))` in the board interpretation; current-state input is an ablation. Mean versus sum scaling should be recorded and kept consistent.

The beta variants reserved in configuration are `none`, `linear_y`, `l2_y`, and `l2_y_z`. The original linear beta term is provisional: minimizing positive `Y` can reward unbounded negative `Y`. Quadratic variants are hypotheses, not a proved fix. Agreement mode may be `none`, `current`, `next`, or `both`; the coefficients are experimental choices. See the [stop-gradient matrix](architecture.md).
