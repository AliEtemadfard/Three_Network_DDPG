"""Strict, deterministic expansion of composable experiment manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import load_config

_ID = re.compile(r"[a-z][a-z0-9_-]*\Z")


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible data independent of mapping order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Variant:
    variant_id: str
    algorithm: str
    environment: str
    experiment: str
    overrides: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    schema_version: int
    name: str
    description: str
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    variants: tuple[Variant, ...]
    output_root: Path
    workers: int
    analysis: dict[str, Any]
    source: dict[str, Any]
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class PlannedRun:
    variant_id: str
    variant_hash: str
    run_id: str
    training_seed: int
    evaluation_seeds: tuple[int, ...]
    config: DictConfig
    config_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_hash": self.variant_hash,
            "run_id": self.run_id,
            "training_seed": self.training_seed,
            "evaluation_seeds": list(self.evaluation_seeds),
            "config_hash": self.config_hash,
            "algorithm": str(self.config.algorithm.name),
            "environment_id": str(self.config.environment.id),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }


def _unique_seeds(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{field} must contain nonnegative integers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicates")
    return result


def load_manifest(path: Path) -> ExperimentManifest:
    """Load and validate a version-2 manifest without creating artifacts."""
    raw_config = OmegaConf.load(path)
    raw = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("experiment manifest must be a mapping")
    allowed = {
        "schema_version",
        "name",
        "description",
        "defaults",
        "training_seeds",
        "evaluation_seeds",
        "variants",
        "execution",
        "analysis",
        "preregistration",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
    if raw.get("schema_version") != 2:
        raise ValueError("schema_version must be 2")
    name = raw.get("name")
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ValueError("manifest name must be a lowercase path-safe identifier")
    training_seeds = _unique_seeds(raw.get("training_seeds"), "training_seeds")
    evaluation_seeds = _unique_seeds(raw.get("evaluation_seeds"), "evaluation_seeds")
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("evaluation seeds must be distinct from training seeds")

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    default_environment = str(defaults.get("environment", "pendulum_v1"))
    default_experiment = str(defaults.get("experiment", "ddpg_baseline"))
    raw_variants = raw.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("variants must be a non-empty list")

    variants: list[Variant] = []
    seen: set[str] = set()
    for item in raw_variants:
        if not isinstance(item, dict):
            raise ValueError("every variant must be a mapping")
        unexpected = set(item) - {"id", "algorithm", "environment", "experiment", "overrides"}
        if unexpected:
            raise ValueError(f"unknown variant fields: {sorted(unexpected)}")
        variant_id = item.get("id")
        if not isinstance(variant_id, str) or not _ID.fullmatch(variant_id):
            raise ValueError("variant id must be a lowercase path-safe identifier")
        if variant_id in seen:
            raise ValueError(f"duplicate variant id: {variant_id}")
        seen.add(variant_id)
        overrides = item.get("overrides", [])
        if not isinstance(overrides, list) or not all(isinstance(value, str) for value in overrides):
            raise ValueError(f"variant {variant_id} overrides must be a list of KEY=VALUE strings")
        if any("beta_regularizer.mode=l2_y_z" in value.replace(" ", "") for value in overrides):
            raise ValueError("l2_y_z is not an available scientific variant")
        variants.append(
            Variant(
                variant_id=variant_id,
                algorithm=str(item.get("algorithm", "")),
                environment=str(item.get("environment", default_environment)),
                experiment=str(item.get("experiment", default_experiment)),
                overrides=tuple(overrides),
            )
        )

    execution = raw.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping")
    unknown_execution = set(execution) - {"output_root", "workers"}
    if unknown_execution:
        raise ValueError(f"unknown execution fields: {sorted(unknown_execution)}")
    workers = execution.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("execution.workers must be a positive integer")
    output_root = Path(str(execution.get("output_root", "runs/protocol-v2")))
    analysis = raw.get("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a mapping")
    unknown_analysis = set(analysis) - {
        "last_k",
        "bootstrap_samples",
        "confidence_level",
        "bootstrap_seed",
    }
    if unknown_analysis:
        raise ValueError(f"unknown analysis fields: {sorted(unknown_analysis)}")
    resolved_analysis = {
        "last_k": analysis.get("last_k", 5),
        "bootstrap_samples": analysis.get("bootstrap_samples", 2000),
        "confidence_level": analysis.get("confidence_level", 0.95),
        "bootstrap_seed": analysis.get("bootstrap_seed", 0),
    }
    if (
        isinstance(resolved_analysis["last_k"], bool)
        or not isinstance(resolved_analysis["last_k"], int)
        or resolved_analysis["last_k"] < 1
    ):
        raise ValueError("analysis.last_k must be a positive integer")
    if (
        isinstance(resolved_analysis["bootstrap_samples"], bool)
        or not isinstance(resolved_analysis["bootstrap_samples"], int)
        or resolved_analysis["bootstrap_samples"] < 1
    ):
        raise ValueError("analysis.bootstrap_samples must be a positive integer")
    if (
        isinstance(resolved_analysis["confidence_level"], bool)
        or not isinstance(resolved_analysis["confidence_level"], int | float)
        or not 0 < float(resolved_analysis["confidence_level"]) < 1
    ):
        raise ValueError("analysis.confidence_level must be in (0, 1)")
    if (
        isinstance(resolved_analysis["bootstrap_seed"], bool)
        or not isinstance(resolved_analysis["bootstrap_seed"], int)
        or resolved_analysis["bootstrap_seed"] < 0
    ):
        raise ValueError("analysis.bootstrap_seed must be a nonnegative integer")
    normalized = json.loads(json.dumps(raw, sort_keys=True))
    return ExperimentManifest(
        schema_version=2,
        name=name,
        description=str(raw.get("description", "")),
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
        variants=tuple(variants),
        output_root=output_root,
        workers=workers,
        analysis=resolved_analysis,
        source=normalized,
        manifest_hash=canonical_hash(normalized),
    )


def expand_manifest(manifest: ExperimentManifest) -> tuple[PlannedRun, ...]:
    """Compose every variant for every paired training seed."""
    planned: list[PlannedRun] = []
    seen_run_ids: set[str] = set()
    for variant in manifest.variants:
        seedless = load_config(
            algorithm=variant.algorithm,
            environment=variant.environment,
            experiment=variant.experiment,
            overrides=variant.overrides,
        )
        if seedless.algorithm.implementation_status == "planned":
            raise ValueError(f"variant {variant.variant_id} uses planned algorithm {variant.algorithm}")
        if str(seedless.algorithm.beta_regularizer.mode) == "l2_y_z":
            raise ValueError("l2_y_z is not an available scientific variant")
        seedless_data = OmegaConf.to_container(seedless, resolve=True)
        assert isinstance(seedless_data, dict)
        algorithm_data = seedless_data["algorithm"]
        assert isinstance(algorithm_data, dict)
        algorithm_data["seed"] = None
        variant_hash = canonical_hash({"variant_id": variant.variant_id, "config": seedless_data})
        for seed in manifest.training_seeds:
            config = OmegaConf.merge(seedless, {"algorithm": {"seed": seed}})
            config_data = OmegaConf.to_container(config, resolve=True)
            config_hash = canonical_hash(config_data)
            identity_hash = canonical_hash(
                {
                    "variant_hash": variant_hash,
                    "training_seed": seed,
                    "evaluation_seeds": manifest.evaluation_seeds,
                }
            )
            run_id = f"{variant.variant_id}-seed-{seed}-{identity_hash[:12]}"
            if run_id in seen_run_ids:
                raise ValueError(f"duplicate run id: {run_id}")
            seen_run_ids.add(run_id)
            planned.append(
                PlannedRun(
                    variant_id=variant.variant_id,
                    variant_hash=variant_hash,
                    run_id=run_id,
                    training_seed=seed,
                    evaluation_seeds=manifest.evaluation_seeds,
                    config=config,
                    config_hash=config_hash,
                )
            )
    return tuple(planned)
