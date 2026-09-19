"""No-learning test doubles verify trainer delegation and evaluation isolation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from three_network_ddpg.core.batches import RolloutBatch, Transition, TransitionBatch
from three_network_ddpg.core.interfaces import (
    ActionSelection,
    AlgorithmCapabilities,
    MetricRecord,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    UpdateResult,
)
from three_network_ddpg.core.replay import ReplayBuffer
from three_network_ddpg.environments import TinyContinuousEnv
from three_network_ddpg.evaluation import evaluate
from three_network_ddpg.training import OffPolicyTrainer, TrainerSettings


class NoLearningOffPolicy(OffPolicyAlgorithm):
    capabilities = AlgorithmCapabilities("test_double", False, True, True, "experimental")

    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self.batches: list[TransitionBatch] = []
        self.deterministic_flags: list[bool] = []
        self.mode_changes: list[bool] = []

    def select_action(self, observation: np.ndarray, *, deterministic: bool) -> ActionSelection:
        self.deterministic_flags.append(deterministic)
        return ActionSelection(np.zeros(1, dtype=np.float32), exploration_applied=False)

    def observe_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def update_from_batch(self, batch: TransitionBatch) -> UpdateResult:
        self.batches.append(batch)
        return UpdateResult(
            metrics=(MetricRecord("test_update", float(len(self.batches)), len(self.batches), "train"),),
            updates_applied=1,
        )

    def set_evaluation_mode(self, enabled: bool) -> None:
        self.mode_changes.append(enabled)

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass

    def restore_checkpoint(self, path: Path) -> None:
        pass


class NoLearningOnPolicy(OnPolicyAlgorithm):
    capabilities = AlgorithmCapabilities("test_a3c", True, False, True, "experimental")

    def select_action(self, observation: np.ndarray, *, deterministic: bool) -> ActionSelection:
        return ActionSelection(np.zeros(1, dtype=np.float32), exploration_applied=False)

    def observe_rollout(self, rollout: RolloutBatch) -> None:
        pass

    def update_from_rollout(self, rollout: RolloutBatch) -> UpdateResult:
        return UpdateResult()

    def set_evaluation_mode(self, enabled: bool) -> None:
        pass

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass

    def restore_checkpoint(self, path: Path) -> None:
        pass


def _replay() -> ReplayBuffer:
    return ReplayBuffer(
        capacity=10,
        observation_shape=(1,),
        action_shape=(1,),
        seed=5,
        bootstrap_on_truncation=True,
        device="cpu",
    )


def _settings() -> TrainerSettings:
    return TrainerSettings(
        total_steps=5,
        warmup_steps=1,
        batch_size=2,
        clip_actions=True,
        bootstrap_on_truncation=True,
    )


def test_trainer_delegates_without_loss_logic() -> None:
    algorithm = NoLearningOffPolicy()
    environment = TinyContinuousEnv(horizon=2)
    replay = _replay()
    try:
        trainer = OffPolicyTrainer(
            algorithm=algorithm, environment=environment, replay=replay, settings=_settings()
        )
        summary = trainer.run(seed=7)
        assert summary.environment_steps == 5
        assert summary.gradient_updates == 4
        assert len(summary.completed_episode_returns) == 2
        assert len(algorithm.transitions) == 5
        assert len(algorithm.batches) == 4
        assert all(batch.batch_size == 2 for batch in algorithm.batches)
        assert sum(transition.truncated for transition in algorithm.transitions) == 2
        assert not any(transition.terminated for transition in algorithm.transitions)
        assert algorithm.deterministic_flags == [False] * 4
        assert algorithm.mode_changes == [False]
    finally:
        environment.close()


def test_on_policy_algorithm_is_rejected_by_replay_trainer() -> None:
    environment = TinyContinuousEnv(horizon=2)
    try:
        with pytest.raises(TypeError, match="OffPolicyAlgorithm"):
            OffPolicyTrainer(
                algorithm=NoLearningOnPolicy(),
                environment=environment,
                replay=_replay(),
                settings=_settings(),
            )
    finally:
        environment.close()


def test_evaluator_requests_noise_free_actions_and_no_updates() -> None:
    algorithm = NoLearningOffPolicy()
    environment = TinyContinuousEnv(horizon=2)
    try:
        result = evaluate(algorithm, environment, episodes=2, seed=12, clip_actions=True)
        assert result.lengths == (2, 2)
        assert algorithm.deterministic_flags == [True] * 4
        assert algorithm.mode_changes == [True, False]
        assert not algorithm.transitions and not algorithm.batches
        assert all(metric.phase == "evaluation" for metric in result.metrics)
    finally:
        environment.close()


def test_planned_algorithm_cannot_enter_trainer() -> None:
    algorithm = NoLearningOffPolicy()
    algorithm.capabilities = AlgorithmCapabilities("placeholder", False, True, True, "planned")
    environment = TinyContinuousEnv(horizon=2)
    try:
        with pytest.raises(ValueError, match="planned"):
            OffPolicyTrainer(
                algorithm=algorithm, environment=environment, replay=_replay(), settings=_settings()
            )
    finally:
        environment.close()
