"""Compose packaged YAML groups and reject incoherent scaffold settings."""

import re
from collections.abc import Iterable
from importlib import resources
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .algorithms.registry import get_algorithm_metadata

_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_AGREEMENT = {"none", "current", "next", "both"}
_BETA = {"none", "linear_y", "l2_y", "l2_y_z"}
_SCHEDULE = {"simultaneous", "alternating", "guarded"}


def _read_yaml(*parts: str) -> DictConfig:
    text = resources.files("three_network_ddpg").joinpath("conf", *parts).read_text(encoding="utf-8")
    loaded = OmegaConf.create(text)
    if not isinstance(loaded, DictConfig):
        raise ValueError(f"configuration is not a mapping: {'/'.join(parts)}")
    return loaded


def _group(group: str, name: str) -> DictConfig:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid {group} group name: {name!r}")
    try:
        return _read_yaml(group, f"{name}.yaml")
    except FileNotFoundError as exc:
        raise ValueError(f"unknown {group} group: {name}") from exc


def load_config(
    *,
    algorithm: str | None = None,
    environment: str | None = None,
    experiment: str | None = None,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Compose algorithm, environment, and experiment groups with OmegaConf."""
    selectors = _read_yaml("config.yaml")
    algorithm_name = algorithm or selectors.algorithm
    environment_name = environment or selectors.environment
    experiment_name = experiment or selectors.experiment
    algorithm_config = OmegaConf.merge(_group("algorithm", "base"), _group("algorithm", algorithm_name))
    config = OmegaConf.create(
        {
            "algorithm": algorithm_config,
            "environment": _group("environment", environment_name),
            "experiment": _group("experiment", experiment_name),
        }
    )
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    validate_config(config)
    return config


def _number(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value > 0 if positive else value >= 0 if nonnegative else True


def validate_config(config: DictConfig) -> None:
    """Validate required fields and architecture-specific constraints."""
    errors: list[str] = []
    try:
        algorithm = config.algorithm
        environment = config.environment
        experiment = config.experiment
        metadata = get_algorithm_metadata(algorithm.name)
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"missing or unknown configuration group: {exc}") from exc

    for field in (
        "implementation_status",
        "is_on_policy",
        "uses_replay_buffer",
        "supports_continuous_actions",
    ):
        if getattr(algorithm, field, None) != getattr(metadata, field):
            errors.append(f"algorithm.{field} must match registry metadata")
    if algorithm.name == "three_network_ddpg" and "target_networks" in algorithm:
        errors.append("proposed main variant must not define target networks")
    if algorithm.name in {"ddpg", "td3"}:
        if "target_networks" not in algorithm:
            errors.append("DDPG and TD3 require explicit target-network settings")
        elif not (algorithm.target_networks.actor and algorithm.target_networks.critic):
            errors.append("DDPG and TD3 target actor and critic must be enabled")
    if algorithm.agreement_mode not in _AGREEMENT:
        errors.append("algorithm.agreement_mode is invalid")
    if algorithm.beta_regularizer.mode not in _BETA:
        errors.append("algorithm.beta_regularizer.mode is invalid")
    if algorithm.update_schedule not in _SCHEDULE:
        errors.append("algorithm.update_schedule is invalid")
    if algorithm.actor_state_source not in {"current", "next"}:
        errors.append("algorithm.actor_state_source is invalid")
    if algorithm.critic_initialization not in {"independent", "identical"}:
        errors.append("algorithm.critic_initialization is invalid")
    if algorithm.device not in {"cpu", "cuda", "auto"}:
        errors.append("algorithm.device is invalid")
    if algorithm.exploration_noise.mode not in {"gaussian", "ou", "policy", "none"}:
        errors.append("algorithm.exploration_noise.mode is invalid")
    if not _number(algorithm.gamma, positive=True) or algorithm.gamma > 1:
        errors.append("algorithm.gamma must be in (0, 1]")
    for name in (
        "actor_learning_rate",
        "critic_one_learning_rate",
        "critic_two_learning_rate",
    ):
        if not _number(getattr(algorithm, name), positive=True):
            errors.append(f"algorithm.{name} must be positive")
    for name in ("alpha", "alpha_prime"):
        if not _number(getattr(algorithm, name), nonnegative=True):
            errors.append(f"algorithm.{name} must be nonnegative")
    if not _number(algorithm.beta_regularizer.coefficient, nonnegative=True):
        errors.append("algorithm.beta_regularizer.coefficient must be nonnegative")
    for name in ("seed", "actor_update_delay", "diagnostic_frequency"):
        value = getattr(algorithm, name)
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f"algorithm.{name} must be an integer >= {minimum}")
    guard = algorithm.guard
    if not _number(guard.tolerance, nonnegative=True):
        errors.append("algorithm.guard.tolerance must be nonnegative")
    if not _number(guard.backtracking_factor, positive=True) or (guard.backtracking_factor >= 1):
        errors.append("algorithm.guard.backtracking_factor must be in (0, 1)")
    if not isinstance(guard.maximum_retries, int) or guard.maximum_retries < 0:
        errors.append("algorithm.guard.maximum_retries must be nonnegative")
    if not _number(guard.minimum_learning_rate, positive=True):
        errors.append("algorithm.guard.minimum_learning_rate must be positive")
    if not isinstance(guard.validation_batch, bool):
        errors.append("algorithm.guard.validation_batch must be boolean")
    if (
        not environment.id
        or not isinstance(environment.wrappers, (list,))
        and not (OmegaConf.is_list(environment.wrappers))
    ):
        errors.append("environment id and wrappers are required")
    for name in (
        "total_steps",
        "batch_size",
        "replay_capacity",
        "evaluation_frequency",
        "evaluation_episodes",
    ):
        value = getattr(experiment, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"experiment.{name} must be a positive integer")
    if experiment.warmup_steps < 0 or experiment.warmup_steps >= experiment.total_steps:
        errors.append("experiment.warmup_steps must be in [0, total_steps)")
    if algorithm.uses_replay_buffer and experiment.batch_size > experiment.replay_capacity:
        errors.append("batch size cannot exceed replay capacity")
    if not isinstance(experiment.bootstrap_on_truncation, bool):
        errors.append("experiment.bootstrap_on_truncation must be boolean")
    if errors:
        raise ValueError("; ".join(errors))
