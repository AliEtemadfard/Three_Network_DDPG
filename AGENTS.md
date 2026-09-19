# Instructions for future coding agents

- Read `docs/research/problem.md` and `docs/research/notation.md` before editing algorithms; consult the original DOCX when interpretation is uncertain.
- Preserve the stop-gradient matrix in `docs/research/architecture.md` exactly. During Critic Two updates only its parameters receive gradients; during Critic One updates only its parameters receive gradients; during actor updates the actor receives gradients through Critic Two's action input while both critics' parameters stay fixed.
- Keep target actor and target critic disabled in the proposed main method unless a named ablation explicitly enables them. Baseline target settings belong to baseline configs.
- Treat `terminated` and `truncated` separately and document the bootstrap mask policy. Never silently collapse them.
- Avoid hidden defaults and magic numbers. Make every experimental choice configurable, including gradient clipping and reward or observation normalization.
- Keep training, evaluation, environment wrappers, algorithms, tracking, and plotting in separate modules.
- Add focused tests for every gradient-path, terminal-mask, or update-order change. Update relevant documentation and `CHANGELOG.md`.
- Do not claim global convergence, optimal-policy convergence, monotonic return improvement, or novelty before evidence supports it.
- Never copy external baseline code without recording source URL, commit or version, license, and local modifications in `THIRD_PARTY_NOTICES.md`.
- Inspect `git diff` and run relevant checks before committing. Never push, force-push, rewrite history, delete user work, or change remotes without explicit approval.
