# ADR 0001: Research scaffold before training

**Status:** Accepted for the bootstrap stage, 2026-09-19.

## Context

The supplied specification fixes the three-network roles but leaves agreement penalties, beta regularization, update schedule, and theory scope open. Implementing training now would silently choose research conclusions and make baseline comparison difficult.

## Decision

Use a Python 3.11 `src/` package, OmegaConf-composed YAML groups, a capability-based algorithm interface, metadata-only registry entries, and CPU-first checks. All six named algorithms are explicit placeholders. Keep proposed-method target networks absent from its main configuration. Preserve the source DOCX unchanged in `docs/research/source/`.

## Consequences

CLI help, configuration inspection, registry metadata, seed utility, and metadata capture can be tested now. No training command exists. DDPG implementation and verification is the next gate. Future variants require named configs, focused gradient tests, and updated evidence before claims are made.
