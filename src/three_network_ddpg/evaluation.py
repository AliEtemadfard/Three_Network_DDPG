"""Evaluation is separate from replay collection and requests noise-free actions."""

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .core.actions import clip_action
from .core.interfaces import Algorithm, MetricRecord
from .core.seeds import seed_environment
from .environments import require_continuous_spaces


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    returns: tuple[float, ...]
    lengths: tuple[int, ...]
    metrics: tuple[MetricRecord, ...]


def evaluate(
    algorithm: Algorithm,
    environment: gym.Env,
    *,
    episodes: int,
    seed: int,
    clip_actions: bool,
    seeds: Sequence[int] | None = None,
) -> EvaluationResult:
    """Evaluate a policy on a separate environment with exploration disabled."""
    evaluation_seeds = tuple(seeds) if seeds is not None else tuple(seed + i for i in range(episodes))
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if len(evaluation_seeds) != episodes or len(set(evaluation_seeds)) != episodes:
        raise ValueError("evaluation seeds must be unique and match the episode count")
    if algorithm.capabilities.implementation_status == "planned":
        raise ValueError("planned algorithms cannot be evaluated")
    require_continuous_spaces(environment)
    returns: list[float] = []
    lengths: list[int] = []
    algorithm.set_evaluation_mode(True)
    try:
        for evaluation_seed in evaluation_seeds:
            observation, _ = seed_environment(environment, evaluation_seed)
            episode_return = 0.0
            length = 0
            while True:
                selection = algorithm.select_action(
                    np.asarray(observation, dtype=np.float32), deterministic=True
                )
                if selection.exploration_applied:
                    raise ValueError("evaluation action contains exploration")
                action = np.asarray(selection.action, dtype=np.float32)
                if clip_actions:
                    action = clip_action(action, environment.action_space.low, environment.action_space.high)
                elif not environment.action_space.contains(action):
                    raise ValueError("algorithm action is outside environment bounds")
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(length)
    finally:
        algorithm.set_evaluation_mode(False)
    metrics: list[MetricRecord] = []
    for index, (episode_return, length) in enumerate(zip(returns, lengths, strict=True)):
        metrics.extend(
            (
                MetricRecord("evaluation_episode_return", episode_return, index, "evaluation"),
                MetricRecord("evaluation_episode_length", float(length), index, "evaluation"),
            )
        )
    metrics.extend(
        (
            MetricRecord("evaluation_return_mean", float(np.mean(returns)), episodes, "evaluation"),
            MetricRecord(
                "evaluation_episode_length_mean",
                float(np.mean(lengths)),
                episodes,
                "evaluation",
            ),
        )
    )
    return EvaluationResult(tuple(returns), tuple(lengths), tuple(metrics))
