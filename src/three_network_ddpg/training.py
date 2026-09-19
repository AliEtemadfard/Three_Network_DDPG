"""Off-policy interaction shell; all update mathematics belongs to the algorithm."""

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if self.total_steps <= 0 or self.batch_size <= 0:
            raise ValueError("total_steps and batch_size must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if not isinstance(self.clip_actions, bool) or not isinstance(self.bootstrap_on_truncation, bool):
            raise TypeError("clip_actions and bootstrap_on_truncation must be boolean")


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    environment_steps: int
    gradient_updates: int
    completed_episode_returns: tuple[float, ...]
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
        self.algorithm = algorithm
        self.environment = environment
        self.replay = replay
        self.settings = settings

    def run(self, *, seed: int, deterministic_algorithms: bool = False) -> TrainingSummary:
        """Collect transitions, sample replay, and delegate updates."""
        seed_everything(seed, deterministic_algorithms=deterministic_algorithms)
        observation, _ = seed_environment(self.environment, seed)
        self.algorithm.set_evaluation_mode(False)
        episode_return = 0.0
        completed_returns: list[float] = []
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
            if step + 1 > self.settings.warmup_steps and len(self.replay) >= self.settings.batch_size:
                result = self.algorithm.update_from_batch(self.replay.sample(self.settings.batch_size))
                if not isinstance(result, UpdateResult):
                    raise TypeError("algorithm update must return UpdateResult")
                gradient_updates += result.updates_applied
                metrics.extend(result.metrics)
            if terminated or truncated:
                completed_returns.append(episode_return)
                episode_return = 0.0
                if step + 1 < self.settings.total_steps:
                    observation, _ = seed_environment(self.environment, seed + len(completed_returns))
            else:
                observation = next_observation
        return TrainingSummary(
            environment_steps=self.settings.total_steps,
            gradient_updates=gradient_updates,
            completed_episode_returns=tuple(completed_returns),
            metrics=tuple(metrics),
        )
