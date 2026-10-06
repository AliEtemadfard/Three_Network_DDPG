"""Compose packaged YAML groups and reject incoherent scaffold settings."""

import re
from collections.abc import Iterable
from importlib import resources
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .algorithms.registry import get_algorithm_metadata

_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_AGREEMENT = {"none", "current", "next", "both"}
_BETA = {"none", "linear_y", "l2_y"}
_CRITIC_OBJECTIVES = {"shared_error", "separate_td"}
_SCHEDULE = {"simultaneous", "sequential", "alternating", "guarded"}


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
    if isinstance(value, bool) or not isinstance(value, int | float):
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
    if algorithm.name == "three_network_ddpg":
        if "target_networks" in algorithm:
            errors.append("proposed main variant must not define target networks")
        if algorithm.critic_objective not in _CRITIC_OBJECTIVES:
            errors.append("three-network critic objective is invalid")
        if algorithm.agreement_mode == "none" and (algorithm.alpha != 0 or algorithm.alpha_prime != 0):
            errors.append("agreement mode none requires zero alpha coefficients")
        if algorithm.agreement_mode == "current" and algorithm.alpha != 0:
            errors.append("current agreement does not use alpha")
        if algorithm.agreement_mode == "next" and algorithm.alpha_prime != 0:
            errors.append("next agreement does not use alpha_prime")
        beta_mode = algorithm.beta_regularizer.mode
        beta_coefficient = algorithm.beta_regularizer.coefficient
        if beta_mode == "none" and beta_coefficient != 0:
            errors.append("beta mode none requires a zero coefficient")
        if beta_mode != "none" and beta_coefficient == 0:
            errors.append("an active beta mode requires a positive coefficient")
        if algorithm.update_schedule not in {"sequential", "simultaneous", "guarded"}:
            errors.append("three-network updates must be sequential, simultaneous, or guarded")
        if algorithm.critic_objective == "separate_td" and algorithm.update_schedule != "sequential":
            errors.append("separate_td requires the sequential update schedule")
        if algorithm.critic_objective == "separate_td" and (beta_mode != "none" or beta_coefficient != 0):
            errors.append("separate_td requires beta regularization to be disabled")
        if algorithm.critic_initialization != "independent":
            errors.append("minimal three-network core requires independent critic initialization")
        if algorithm.actor_update_delay != 1:
            errors.append("minimal three-network core updates the actor every learner step")
        if algorithm.optimizer.name != "adam":
            errors.append("minimal three-network core currently supports the Adam optimizer")
        if not algorithm.action_rescaling:
            errors.append("minimal three-network core requires action rescaling")
        if algorithm.exploration_noise.mode not in {"gaussian", "none"}:
            errors.append("minimal three-network core supports gaussian or disabled exploration")
    if algorithm.name in {"ddpg", "ddpg_no_target_actor", "td3"}:
        if "target_networks" not in algorithm:
            errors.append("DDPG-family and TD3 configs require explicit target-network settings")
        else:
            targets = algorithm.target_networks
            if algorithm.name in {"ddpg", "td3"} and not (targets.actor and targets.critic):
                errors.append("standard DDPG and TD3 target actor and critic must be enabled")
            if algorithm.name == "ddpg_no_target_actor" and (targets.actor or not targets.critic):
                errors.append("ddpg_no_target_actor requires actor=false and critic=true")
            if not _number(targets.tau, positive=True) or targets.tau > 1:
                errors.append("target-network tau must be in (0, 1]")
    if algorithm.name in {"ddpg", "ddpg_no_target_actor"}:
        if algorithm.optimizer.name != "adam":
            errors.append("DDPG baselines currently support the Adam optimizer")
        if not algorithm.action_rescaling:
            errors.append("DDPG baselines require action rescaling")
        if algorithm.actor_update_delay != 1:
            errors.append("DDPG baselines update the actor every update")
        if algorithm.exploration_noise.mode not in {"gaussian", "none"}:
            errors.append("DDPG exploration noise mode must be gaussian or none")
        noise_clip = algorithm.exploration_noise.clip
        if noise_clip is not None and not _number(noise_clip, positive=True):
            errors.append("exploration noise clip must be positive when configured")
    if algorithm.agreement_mode not in _AGREEMENT:
        errors.append("algorithm.agreement_mode is invalid")
    beta_mode = algorithm.beta_regularizer.mode
    if beta_mode == "l2_y_z":
        errors.append(
            "l2_y_z is unavailable because symmetric Y/Z regularization requires a "
            "separately approved objective"
        )
    elif beta_mode not in _BETA:
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
    noise_standard_deviation = algorithm.exploration_noise.standard_deviation
    if noise_standard_deviation is not None and not _number(noise_standard_deviation, nonnegative=True):
        errors.append("algorithm.exploration_noise.standard_deviation must be nonnegative")
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
    for name in ("seed", "actor_update_delay"):
        value = getattr(algorithm, name)
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f"algorithm.{name} must be an integer >= {minimum}")
    diagnostic_frequency = algorithm.diagnostic_frequency
    if diagnostic_frequency is not None and (
        isinstance(diagnostic_frequency, bool)
        or not isinstance(diagnostic_frequency, int)
        or diagnostic_frequency < 1
    ):
        errors.append("algorithm.diagnostic_frequency must be null or a positive integer")
    guard = algorithm.guard
    if guard.scope not in {"joint", "critics", "actor"}:
        errors.append("algorithm.guard.scope must be joint, critics, or actor")
    if not _number(guard.tolerance, nonnegative=True):
        errors.append("algorithm.guard.tolerance must be nonnegative")
    if not _number(guard.backtracking_factor, positive=True) or (guard.backtracking_factor >= 1):
        errors.append("algorithm.guard.backtracking_factor must be in (0, 1)")
    if not isinstance(guard.maximum_retries, int) or guard.maximum_retries < 0:
        errors.append("algorithm.guard.maximum_retries must be nonnegative")
    if not _number(guard.minimum_learning_rate, positive=True):
        errors.append("algorithm.guard.minimum_learning_rate must be positive")
    else:
        learning_rates = (
            algorithm.actor_learning_rate,
            algorithm.critic_one_learning_rate,
            algorithm.critic_two_learning_rate,
        )
        if all(_number(rate, positive=True) for rate in learning_rates) and (
            guard.minimum_learning_rate > min(learning_rates)
        ):
            errors.append("algorithm.guard.minimum_learning_rate exceeds a learning rate")
    if not isinstance(guard.validation_batch, bool):
        errors.append("algorithm.guard.validation_batch must be boolean")
    if (
        not environment.id
        or not isinstance(environment.wrappers, list)
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
    if not isinstance(experiment.clip_actions, bool):
        errors.append("experiment.clip_actions must be boolean")
    if environment.id == "ThreeNetworkTinyContinuous-v0":
        horizon = environment.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            errors.append("environment.horizon must be a positive integer")
    if errors:
        raise ValueError("; ".join(errors))
