"""Focused tests for the controlled DDPG baseline without a target actor."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from three_network_ddpg.algorithms.ddpg import DDPG, DDPGNoTargetActor, DDPGSettings
from three_network_ddpg.algorithms.registry import create_algorithm
from three_network_ddpg.config import load_config, validate_config
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
        "gamma": 0.9,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "tau": 0.25,
        "exploration_noise_std": 0.2,
        "device": "cpu",
        "seed": 13,
    }
    values.update(changes)
    return DDPGSettings(**values)


def _batch(*, mask: torch.Tensor | None = None) -> TransitionBatch:
    generator = torch.Generator().manual_seed(17)
    size = 4
    flags = torch.zeros((size, 1), dtype=torch.bool)
    return TransitionBatch(
        observations=torch.randn((size, 3), generator=generator),
        actions=torch.empty((size, 1)).uniform_(-2.0, 2.0, generator=generator),
        rewards=torch.arange(1, size + 1, dtype=torch.float32).unsqueeze(1),
        next_observations=torch.randn((size, 3), generator=generator),
        terminated=flags,
        truncated=flags.clone(),
        bootstrap_mask=torch.ones((size, 1)) if mask is None else mask,
    )


def _configured_algorithm(*overrides: str) -> DDPGNoTargetActor:
    config = load_config(
        algorithm="ddpg_no_target_actor",
        experiment="ddpg_no_target_actor",
        overrides=["algorithm.hidden_layers=[8,8]", *overrides],
    )
    algorithm = create_algorithm(
        "ddpg_no_target_actor",
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0], dtype=np.float32),
        action_high=np.array([2.0], dtype=np.float32),
    )
    assert isinstance(algorithm, DDPGNoTargetActor)
    return algorithm


def test_registry_config_and_checkpoint_have_no_target_actor() -> None:
    config = load_config(
        algorithm="ddpg_no_target_actor",
        experiment="ddpg_no_target_actor",
    )
    assert config.algorithm.name == "ddpg_no_target_actor"
    assert config.algorithm.implementation_status == "experimental"
    assert not config.algorithm.target_networks.actor
    assert config.algorithm.target_networks.critic

    algorithm = _configured_algorithm()
    assert algorithm.capabilities.name == "ddpg_no_target_actor"
    assert algorithm.target_actor is None
    assert "target_actor" not in algorithm.state_dict()


def test_config_rejects_any_target_actor_or_missing_target_critic() -> None:
    config = load_config(
        algorithm="ddpg_no_target_actor",
        experiment="ddpg_no_target_actor",
    )
    config.algorithm.target_networks.actor = True
    with pytest.raises(ValueError, match="actor=false and critic=true"):
        validate_config(config)
    with pytest.raises(ValueError, match="actor=false.*critic=true"):
        DDPGNoTargetActor.from_config(
            config.algorithm,
            observation_dim=3,
            action_low=np.array([-2.0], dtype=np.float32),
            action_high=np.array([2.0], dtype=np.float32),
        )

    config.algorithm.target_networks.actor = False
    config.algorithm.target_networks.critic = False
    with pytest.raises(ValueError, match="actor=false and critic=true"):
        validate_config(config)


def test_critic_target_uses_online_actor_target_critic_and_explicit_mask() -> None:
    algorithm = DDPGNoTargetActor(_settings())
    mask = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    batch = _batch(mask=mask)

    actual = algorithm._critic_targets(batch)
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        next_q = algorithm.target_critic(batch.next_observations, next_actions)
        expected = batch.rewards + algorithm.settings.gamma * mask * next_q

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[mask == 0], batch.rewards[mask == 0])
    assert not actual.requires_grad


def test_critic_regression_cannot_create_actor_gradients() -> None:
    algorithm = DDPGNoTargetActor(_settings())
    batch = _batch()
    algorithm.actor_optimizer.zero_grad(set_to_none=True)
    algorithm.critic_optimizer.zero_grad(set_to_none=True)

    targets = algorithm._critic_targets(batch)
    predicted = algorithm.critic(batch.observations, batch.actions)
    torch.nn.functional.mse_loss(predicted, targets).backward()

    assert all(parameter.grad is None for parameter in algorithm.actor.parameters())
    assert any(parameter.grad is not None for parameter in algorithm.critic.parameters())
    assert all(parameter.grad is None for parameter in algorithm.target_critic.parameters())


def test_soft_update_changes_only_target_critic() -> None:
    algorithm = DDPGNoTargetActor(_settings(tau=0.25))
    actor_before = deepcopy(algorithm.actor.state_dict())
    with torch.no_grad():
        for parameter in algorithm.critic.parameters():
            parameter.fill_(2.0)
        for parameter in algorithm.target_critic.parameters():
            parameter.zero_()

    algorithm._soft_update_targets()

    for parameter in algorithm.target_critic.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))
    for name, value in actor_before.items():
        torch.testing.assert_close(value, algorithm.actor.state_dict()[name])
    assert algorithm.target_actor is None


def test_checkpoint_round_trip_has_distinct_identity_and_no_target_actor(tmp_path: Path) -> None:
    source = DDPGNoTargetActor(_settings())
    source.update_from_batch(_batch())
    checkpoint = tmp_path / "ddpg_no_target_actor.pt"
    source.save_checkpoint(checkpoint)

    restored = DDPGNoTargetActor(_settings(seed=99))
    restored.restore_checkpoint(checkpoint)
    expected = source.state_dict()
    actual = restored.state_dict()

    assert actual["algorithm"] == "ddpg_no_target_actor"
    assert "target_actor" not in actual
    for network in ("actor", "critic", "target_critic"):
        for key, value in expected[network].items():
            torch.testing.assert_close(value, actual[network][key])
    assert actual["update_step"] == expected["update_step"]

    with pytest.raises(ValueError, match="checkpoint algorithm"):
        DDPG(_settings()).load_state_dict(actual)


def test_evaluation_is_deterministic_and_noise_free() -> None:
    algorithm = _configured_algorithm(
        "algorithm.exploration_noise.mode=gaussian",
        "algorithm.exploration_noise.standard_deviation=0.8",
    )
    observation = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    exploratory = algorithm.select_action(observation, deterministic=False)
    assert exploratory.exploration_applied

    algorithm.set_evaluation_mode(True)
    first = algorithm.select_action(observation, deterministic=False)
    second = algorithm.select_action(observation, deterministic=True)
    assert not first.exploration_applied
    assert not second.exploration_applied
    np.testing.assert_array_equal(first.action, second.action)


@pytest.mark.parametrize(
    ("environment_group", "steps", "warmup", "batch_size", "expected_length"),
    [
        ("tiny_continuous", 16, 4, 4, 5),
        ("pendulum_v1", 24, 8, 8, 200),
    ],
)
def test_short_training_and_evaluation_smoke(
    environment_group: str,
    steps: int,
    warmup: int,
    batch_size: int,
    expected_length: int,
) -> None:
    config = load_config(
        algorithm="ddpg_no_target_actor",
        environment=environment_group,
        experiment="ddpg_no_target_actor",
        overrides=[
            "algorithm.hidden_layers=[8,8]",
            "algorithm.seed=23",
            f"experiment.total_steps={steps}",
            f"experiment.warmup_steps={warmup}",
            f"experiment.batch_size={batch_size}",
            "experiment.replay_capacity=64",
            "experiment.evaluation_episodes=1",
        ],
    )
    environment = make_environment(config.environment)
    evaluation_environment = make_environment(config.environment)
    try:
        observation_shape = environment.observation_space.shape
        action_shape = environment.action_space.shape
        assert observation_shape is not None and action_shape is not None
        algorithm = DDPGNoTargetActor.from_config(
            config.algorithm,
            observation_dim=int(observation_shape[0]),
            action_low=environment.action_space.low,
            action_high=environment.action_space.high,
        )
        replay = ReplayBuffer(
            capacity=64,
            observation_shape=observation_shape,
            action_shape=action_shape,
            seed=23,
            bootstrap_on_truncation=True,
            device="cpu",
        )
        summary = OffPolicyTrainer(
            algorithm=algorithm,
            environment=environment,
            replay=replay,
            settings=TrainerSettings(steps, warmup, batch_size, True, True),
        ).run(seed=23, deterministic_algorithms=True)
        evaluation = evaluate(
            algorithm,
            evaluation_environment,
            episodes=1,
            seed=1023,
            clip_actions=True,
        )

        assert summary.environment_steps == steps
        assert summary.gradient_updates == steps - warmup
        assert evaluation.lengths == (expected_length,)
        assert algorithm.target_actor is None
    finally:
        environment.close()
        evaluation_environment.close()
