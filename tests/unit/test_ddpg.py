"""Focused correctness tests for the standard four-network DDPG baseline."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from three_network_ddpg.algorithms.ddpg import (
    DDPG,
    DDPGSettings,
    DeterministicActor,
    QCritic,
    ddpg_td_target,
    soft_update,
)
from three_network_ddpg.core.batches import TransitionBatch
from three_network_ddpg.core.replay import ReplayBuffer
from three_network_ddpg.environments import make_environment
from three_network_ddpg.evaluation import evaluate
from three_network_ddpg.training import OffPolicyTrainer, TrainerSettings


def _settings(**changes: object) -> DDPGSettings:
    values: dict[str, object] = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (16, 16),
        "activation": "relu",
        "gamma": 0.99,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "tau": 0.01,
        "exploration_noise_std": 0.2,
        "device": "cpu",
        "seed": 7,
    }
    values.update(changes)
    return DDPGSettings(**values)


def _batch(
    *,
    batch_size: int = 8,
    mask: torch.Tensor | None = None,
    reward: float = 1.0,
) -> TransitionBatch:
    generator = torch.Generator().manual_seed(3)
    observations = torch.randn((batch_size, 3), generator=generator)
    actions = torch.empty((batch_size, 1)).uniform_(-2.0, 2.0, generator=generator)
    next_observations = torch.randn((batch_size, 3), generator=generator)
    terminated = torch.zeros((batch_size, 1), dtype=torch.bool)
    truncated = torch.zeros((batch_size, 1), dtype=torch.bool)
    bootstrap = torch.ones((batch_size, 1)) if mask is None else mask
    return TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=torch.full((batch_size, 1), reward),
        next_observations=next_observations,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=bootstrap,
    )


def test_network_shapes_action_bounds_and_initial_target_copies() -> None:
    settings = _settings()
    actor = DeterministicActor(settings)
    critic = QCritic(settings)
    observations = torch.randn(5, 3)
    actions = actor(observations)
    assert actions.shape == (5, 1)
    assert torch.all(actions >= -2.0) and torch.all(actions <= 2.0)
    assert critic(observations, actions).shape == (5, 1)

    algorithm = DDPG(settings)
    for online, target in zip(algorithm.actor.parameters(), algorithm.target_actor.parameters(), strict=True):
        torch.testing.assert_close(online, target)
        assert not target.requires_grad
    for online, target in zip(
        algorithm.critic.parameters(), algorithm.target_critic.parameters(), strict=True
    ):
        torch.testing.assert_close(online, target)
        assert not target.requires_grad


def test_td_target_uses_terminal_and_truncation_policy_mask() -> None:
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    next_q = torch.tensor([[10.0], [20.0], [30.0]])
    mask_bootstrap_truncation = torch.tensor([[0.0], [1.0], [1.0]])
    mask_stop_truncation = torch.tensor([[0.0], [0.0], [1.0]])
    torch.testing.assert_close(
        ddpg_td_target(rewards, mask_bootstrap_truncation, next_q, 0.5),
        torch.tensor([[1.0], [11.0], [16.0]]),
    )
    torch.testing.assert_close(
        ddpg_td_target(rewards, mask_stop_truncation, next_q, 0.5),
        torch.tensor([[1.0], [1.0], [16.0]]),
    )


def test_soft_update_matches_polyak_equation() -> None:
    online = torch.nn.Linear(2, 1)
    target = deepcopy(online)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(2.0)
        for parameter in target.parameters():
            parameter.zero_()
    soft_update(target, online, 0.25)
    for parameter in target.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))


def test_actor_gradient_flows_while_critic_is_isolated() -> None:
    algorithm = DDPG(_settings())
    actor_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    result = algorithm.update_from_batch(_batch())
    assert result.updates_applied == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, algorithm.actor.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in algorithm.critic.parameters())
    assert all(parameter.grad is None for parameter in algorithm.target_actor.parameters())
    assert all(parameter.grad is None for parameter in algorithm.target_critic.parameters())
    metrics = {metric.name: metric.value for metric in result.metrics}
    assert metrics["actor_gradient_norm"] > 0
    assert metrics["critic_gradient_norm"] > 0


def test_checkpoint_round_trip_restores_all_four_networks_and_counters(
    tmp_path: Path,
) -> None:
    algorithm = DDPG(_settings())
    algorithm.update_from_batch(_batch())
    checkpoint = tmp_path / "ddpg.pt"
    algorithm.save_checkpoint(checkpoint)
    expected = deepcopy(algorithm.state_dict())

    restored = DDPG(_settings(seed=99))
    restored.restore_checkpoint(checkpoint)
    actual = restored.state_dict()
    for network in ("actor", "critic", "target_actor", "target_critic"):
        for key, value in expected[network].items():
            torch.testing.assert_close(value, actual[network][key])
    assert actual["update_step"] == expected["update_step"]
    assert actual["environment_steps"] == expected["environment_steps"]


def test_deterministic_evaluation_has_no_exploration_noise() -> None:
    algorithm = DDPG(_settings(exploration_noise_std=0.5))
    observation = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    first = algorithm.select_action(observation, deterministic=True)
    second = algorithm.select_action(observation, deterministic=True)
    np.testing.assert_array_equal(first.action, second.action)
    assert not first.exploration_applied and not second.exploration_applied
    exploratory = algorithm.select_action(observation, deterministic=False)
    assert exploratory.exploration_applied
    assert not np.array_equal(first.action, exploratory.action)


def test_repeated_deterministic_transition_reduces_critic_loss() -> None:
    algorithm = DDPG(
        _settings(
            gamma=0.99,
            actor_learning_rate=1e-5,
            critic_learning_rate=5e-3,
            tau=0.01,
        )
    )
    batch = _batch(batch_size=32, mask=torch.zeros((32, 1)), reward=1.5)
    losses = []
    for _ in range(50):
        result = algorithm.update_from_batch(batch)
        losses.append(next(metric.value for metric in result.metrics if metric.name == "critic_loss"))
    assert losses[-1] < losses[0] * 0.2


def test_short_pendulum_training_and_evaluation_smoke() -> None:
    import three_network_ddpg.config as project_config

    config = project_config.load_config(
        algorithm="ddpg",
        environment="pendulum_v1",
        experiment="ddpg_baseline",
        overrides=[
            "algorithm.hidden_layers=[16,16]",
            "algorithm.seed=5",
            "experiment.total_steps=24",
            "experiment.warmup_steps=8",
            "experiment.batch_size=8",
            "experiment.replay_capacity=64",
            "experiment.evaluation_episodes=1",
        ],
    )
    environment = make_environment(config.environment)
    evaluation_environment = make_environment(config.environment)
    try:
        algorithm = DDPG.from_config(
            config.algorithm,
            observation_dim=3,
            action_low=environment.action_space.low,
            action_high=environment.action_space.high,
        )
        replay = ReplayBuffer(
            capacity=64,
            observation_shape=(3,),
            action_shape=(1,),
            seed=5,
            bootstrap_on_truncation=True,
            device="cpu",
        )
        summary = OffPolicyTrainer(
            algorithm=algorithm,
            environment=environment,
            replay=replay,
            settings=TrainerSettings(24, 8, 8, True, True),
        ).run(seed=5, deterministic_algorithms=True)
        result = evaluate(
            algorithm,
            evaluation_environment,
            episodes=1,
            seed=1005,
            clip_actions=True,
        )
        assert summary.environment_steps == 24
        assert summary.gradient_updates == 16
        assert summary.wall_time_seconds > 0
        required = {
            "critic_loss",
            "actor_loss",
            "q_mean",
            "q_abs_max",
            "critic_gradient_norm",
            "actor_gradient_norm",
            "environment_steps",
            "update_steps",
            "wall_time_seconds",
        }
        assert required <= {metric.name for metric in summary.metrics}
        assert result.lengths == (200,)
        assert {
            "evaluation_episode_return",
            "evaluation_episode_length",
            "evaluation_return_mean",
        } <= {metric.name for metric in result.metrics}
    finally:
        environment.close()
        evaluation_environment.close()
