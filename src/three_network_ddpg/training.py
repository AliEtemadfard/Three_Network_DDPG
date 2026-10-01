"""Off-policy interaction shell; all update mathematics belongs to the algorithm."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import gymnasium as gym
import numpy as np

from .core.actions import clip_action
from .core.batches import Transition
from .core.interfaces import MetricRecord, OffPolicyAlgorithm, UpdateResult
from .core.replay import ReplayBuffer
from .core.seeds import seed_environment, seed_everything
from .environments import require_continuous_spaces


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    total_steps: int
    warmup_steps: int
    batch_size: int
    clip_actions: bool
    bootstrap_on_truncation: bool
    evaluation_frequency: int | None = None

    def __post_init__(self) -> None:
        if self.total_steps <= 0 or self.batch_size <= 0:
            raise ValueError("total_steps and batch_size must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if not isinstance(self.clip_actions, bool) or not isinstance(self.bootstrap_on_truncation, bool):
            raise TypeError("clip_actions and bootstrap_on_truncation must be boolean")
        if self.evaluation_frequency is not None and self.evaluation_frequency <= 0:
            raise ValueError("evaluation_frequency must be positive when configured")


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    environment_steps: int
    gradient_updates: int
    completed_episode_returns: tuple[float, ...]
    completed_episode_lengths: tuple[int, ...]
    wall_time_seconds: float
    metrics: tuple[MetricRecord, ...]


class OffPolicyTrainer:
    """Replay-driven loop restricted to an implemented off-policy algorithm."""

    def __init__(
        self,
        *,
        algorithm: OffPolicyAlgorithm,
        environment: gym.Env,
        replay: ReplayBuffer,
        settings: TrainerSettings,
        evaluation_callback: Callable[[int, OffPolicyAlgorithm], tuple[MetricRecord, ...]] | None = None,
        metric_sink: Callable[[MetricRecord], None] | None = None,
    ) -> None:
        if not isinstance(algorithm, OffPolicyAlgorithm):
            raise TypeError("off-policy trainer requires OffPolicyAlgorithm")
        capabilities = algorithm.capabilities
        if capabilities.is_on_policy or not capabilities.uses_replay_buffer:
            raise ValueError("algorithm capabilities do not permit replay training")
        if not capabilities.supports_continuous_actions:
            raise ValueError("algorithm must support continuous actions")
        if capabilities.implementation_status == "planned":
            raise ValueError("planned algorithms cannot enter the trainer")
        require_continuous_spaces(environment)
        if environment.observation_space.shape != replay.observation_shape:
            raise ValueError("environment observation shape differs from replay")
        if environment.action_space.shape != replay.action_shape:
            raise ValueError("environment action shape differs from replay")
        if replay.bootstrap_on_truncation != settings.bootstrap_on_truncation:
            raise ValueError("replay and trainer truncation policies differ")
        if settings.batch_size > replay.capacity:
            raise ValueError("batch_size exceeds replay capacity")
        if (settings.evaluation_frequency is None) != (evaluation_callback is None):
            raise ValueError("evaluation frequency and callback must be configured together")
        self.algorithm = algorithm
        self.environment = environment
        self.replay = replay
        self.settings = settings
        self.evaluation_callback = evaluation_callback
        self.metric_sink = metric_sink

    def _record(self, metrics: list[MetricRecord], records: tuple[MetricRecord, ...]) -> None:
        metrics.extend(records)
        if self.metric_sink is not None:
            for record in records:
                self.metric_sink(record)

    def run(self, *, seed: int, deterministic_algorithms: bool = False) -> TrainingSummary:
        """Collect transitions, sample replay, and delegate updates."""
        started = perf_counter()
        seed_everything(seed, deterministic_algorithms=deterministic_algorithms)
        observation, _ = seed_environment(self.environment, seed)
        self.algorithm.set_evaluation_mode(False)
        episode_return = 0.0
        episode_length = 0
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        metrics: list[MetricRecord] = []
        gradient_updates = 0
        for step in range(self.settings.total_steps):
            if step < self.settings.warmup_steps:
                action = self.environment.action_space.sample()
            else:
                selection = self.algorithm.select_action(
                    np.asarray(observation, dtype=np.float32), deterministic=False
                )
                action = selection.action
            action = np.asarray(action, dtype=np.float32)
            if self.settings.clip_actions:
                action = clip_action(
                    action, self.environment.action_space.low, self.environment.action_space.high
                )
            elif not self.environment.action_space.contains(action):
                raise ValueError("algorithm action is outside environment bounds")
            next_observation, reward, terminated, truncated, _ = self.environment.step(action)
            transition = Transition(
                observation=np.asarray(observation, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32),
                terminated=terminated,
                truncated=truncated,
            )
            self.replay.add(transition)
            self.algorithm.observe_transition(transition)
            episode_return += float(reward)
            episode_length += 1
            if step + 1 > self.settings.warmup_steps and len(self.replay) >= self.settings.batch_size:
                training_batch = self.replay.sample(self.settings.batch_size)
                validation_batch = (
                    self.replay.sample(self.settings.batch_size)
                    if self.algorithm.requires_validation_batch
                    else None
                )
                result = self.algorithm.update_from_batch(
                    training_batch,
                    validation_batch=validation_batch,
                )
                if not isinstance(result, UpdateResult):
                    raise TypeError("algorithm update must return UpdateResult")
                gradient_updates += result.updates_applied
                self._record(metrics, result.metrics)
            if terminated or truncated:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                self._record(
                    metrics,
                    (
                        MetricRecord("episode_return", episode_return, step + 1, "train"),
                        MetricRecord("episode_length", float(episode_length), step + 1, "train"),
                    ),
                )
                episode_return = 0.0
                episode_length = 0
                if step + 1 < self.settings.total_steps:
                    observation, _ = seed_environment(self.environment, seed + len(completed_returns))
            else:
                observation = next_observation
            if (
                self.evaluation_callback is not None
                and self.settings.evaluation_frequency is not None
                and (step + 1) % self.settings.evaluation_frequency == 0
            ):
                self._record(metrics, self.evaluation_callback(step + 1, self.algorithm))
        wall_time = perf_counter() - started
        self._record(
            metrics,
            (
                MetricRecord(
                    "environment_steps", float(self.settings.total_steps), self.settings.total_steps, "train"
                ),
                MetricRecord("update_steps", float(gradient_updates), self.settings.total_steps, "train"),
                MetricRecord("wall_time_seconds", wall_time, self.settings.total_steps, "train"),
            ),
        )
        return TrainingSummary(
            environment_steps=self.settings.total_steps,
            gradient_updates=gradient_updates,
            completed_episode_returns=tuple(completed_returns),
            completed_episode_lengths=tuple(completed_lengths),
            wall_time_seconds=wall_time,
            metrics=tuple(metrics),
        )
