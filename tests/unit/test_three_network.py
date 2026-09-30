"""Focused gradient and execution tests for the minimal three-network method."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from three_network_ddpg.algorithms.three_network_ddpg import (
    ThreeNetworkDDPG,
    ThreeNetworkSettings,
    shared_residual,
    shared_squared_error,
)
from three_network_ddpg.config import load_config
from three_network_ddpg.core.batches import TransitionBatch
from three_network_ddpg.core.replay import ReplayBuffer
from three_network_ddpg.environments import make_environment
from three_network_ddpg.evaluation import evaluate
from three_network_ddpg.training import OffPolicyTrainer, TrainerSettings


def _settings(**changes: Any) -> ThreeNetworkSettings:
    values: dict[str, Any] = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (16, 16),
        "activation": "relu",
        "gamma": 0.95,
        "actor_learning_rate": 1e-3,
        "critic_one_learning_rate": 1e-3,
        "critic_two_learning_rate": 1e-3,
        "update_schedule": "sequential",
        "actor_state_source": "next",
        "exploration_noise_std": 0.2,
        "diagnostic_frequency": 1,
        "device": "cpu",
        "seed": 13,
    }
    values.update(changes)
    return ThreeNetworkSettings(**values)


def _batch(batch_size: int = 8) -> TransitionBatch:
    generator = torch.Generator().manual_seed(9)
    terminated = torch.zeros((batch_size, 1), dtype=torch.bool)
    truncated = torch.zeros((batch_size, 1), dtype=torch.bool)
    return TransitionBatch(
        observations=torch.randn((batch_size, 3), generator=generator),
        actions=torch.empty((batch_size, 1)).uniform_(-2.0, 2.0, generator=generator),
        rewards=torch.randn((batch_size, 1), generator=generator),
        next_observations=torch.randn((batch_size, 3), generator=generator),
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=torch.ones((batch_size, 1)),
    )


def _copies(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(previous, current)
        for previous, current in zip(before, module.parameters(), strict=True)
    )


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


def test_shared_residual_and_half_squared_error_use_explicit_mask() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    mask = torch.tensor([[0.0], [1.0]])
    z = torch.tensor([[20.0], [4.0]])
    y = torch.tensor([[3.0], [2.0]])
    delta = shared_residual(rewards, mask, z, y, gamma=0.5)
    torch.testing.assert_close(delta, torch.tensor([[-2.0], [1.0]]))
    torch.testing.assert_close(shared_squared_error(delta), torch.tensor(1.25))


def test_critics_are_independent_and_no_target_modules_or_state_exist() -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    critic_one = tuple(algorithm.critic_one.parameters())
    critic_two = tuple(algorithm.critic_two.parameters())
    assert all(one is not two for one, two in zip(critic_one, critic_two, strict=True))
    assert any(not torch.equal(one, two) for one, two in zip(critic_one, critic_two, strict=True))
    assert not hasattr(algorithm, "target_actor")
    assert not hasattr(algorithm, "target_critic")
    module_attributes = {
        name for name, value in vars(algorithm).items() if isinstance(value, torch.nn.Module)
    }
    assert module_attributes == {"actor", "critic_one", "critic_two"}
    assert all("target" not in name for name in module_attributes)
    assert not {"target_actor", "target_critic"} & set(algorithm.state_dict())


def test_critic_two_graph_reaches_only_critic_two_parameters() -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    batch = _batch()
    algorithm._zero_all_gradients()
    actions = algorithm.actor(batch.next_observations).detach()
    y = algorithm.critic_one(batch.observations, batch.actions).detach()
    z = algorithm.critic_two(batch.next_observations, actions)
    shared_squared_error(
        shared_residual(batch.rewards, batch.bootstrap_mask, z, y, algorithm.settings.gamma)
    ).backward()
    assert not _has_gradient(algorithm.actor)
    assert not _has_gradient(algorithm.critic_one)
    assert _has_gradient(algorithm.critic_two)


def test_critic_one_graph_reaches_only_critic_one_parameters() -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    batch = _batch()
    algorithm._zero_all_gradients()
    actions = algorithm.actor(batch.next_observations).detach()
    z = algorithm.critic_two(batch.next_observations, actions).detach()
    y = algorithm.critic_one(batch.observations, batch.actions)
    shared_squared_error(
        shared_residual(batch.rewards, batch.bootstrap_mask, z, y, algorithm.settings.gamma)
    ).backward()
    assert not _has_gradient(algorithm.actor)
    assert _has_gradient(algorithm.critic_one)
    assert not _has_gradient(algorithm.critic_two)


def test_actor_graph_freezes_critics_but_retains_action_path() -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    batch = _batch()
    algorithm._zero_all_gradients()
    critic_parameters = tuple(algorithm.critic_one.parameters()) + tuple(algorithm.critic_two.parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    actions = algorithm.actor(batch.next_observations)
    output = algorithm.critic_two(batch.next_observations, actions)
    assert output.grad_fn is not None
    (-output.mean()).backward()
    assert _has_gradient(algorithm.actor)
    assert not _has_gradient(algorithm.critic_one)
    assert not _has_gradient(algorithm.critic_two)
    for parameter in critic_parameters:
        parameter.requires_grad_(True)


def test_sequential_phases_mutate_only_their_parameter_group() -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    batch = _batch()

    actor_before = _copies(algorithm.actor)
    one_before = _copies(algorithm.critic_one)
    two_before = _copies(algorithm.critic_two)
    algorithm._sequential_critic_two(batch)
    assert not _changed(actor_before, algorithm.actor)
    assert not _changed(one_before, algorithm.critic_one)
    assert _changed(two_before, algorithm.critic_two)

    actor_before = _copies(algorithm.actor)
    one_before = _copies(algorithm.critic_one)
    two_before = _copies(algorithm.critic_two)
    algorithm._sequential_critic_one(batch)
    assert not _changed(actor_before, algorithm.actor)
    assert _changed(one_before, algorithm.critic_one)
    assert not _changed(two_before, algorithm.critic_two)

    actor_before = _copies(algorithm.actor)
    one_before = _copies(algorithm.critic_one)
    two_before = _copies(algorithm.critic_two)
    algorithm._sequential_actor(batch)
    assert _changed(actor_before, algorithm.actor)
    assert not _changed(one_before, algorithm.critic_one)
    assert not _changed(two_before, algorithm.critic_two)


def test_simultaneous_stores_every_gradient_before_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(_settings(update_schedule="simultaneous"))
    optimizers = (
        algorithm.critic_two_optimizer,
        algorithm.critic_one_optimizer,
        algorithm.actor_optimizer,
    )
    first_step_checked = False
    order: list[int] = []

    for index, optimizer in enumerate(optimizers):
        original_step = optimizer.step

        def wrapped_step(
            *args: Any,
            _index: int = index,
            _original: Any = original_step,
            **kwargs: Any,
        ) -> Any:
            nonlocal first_step_checked
            if not first_step_checked:
                assert all(
                    parameter.grad is not None
                    for module in (
                        algorithm.actor,
                        algorithm.critic_one,
                        algorithm.critic_two,
                    )
                    for parameter in module.parameters()
                )
                first_step_checked = True
            order.append(_index)
            return _original(*args, **kwargs)

        monkeypatch.setattr(optimizer, "step", wrapped_step)

    result = algorithm.update_from_batch(_batch())
    assert result.updates_applied == 1
    assert first_step_checked
    assert order == [0, 1, 2]


@pytest.mark.parametrize("schedule", ["sequential", "simultaneous"])
def test_update_logs_required_shared_error_statistics_and_diagnostics(
    schedule: str,
) -> None:
    algorithm = ThreeNetworkDDPG(_settings(update_schedule=schedule))
    result = algorithm.update_from_batch(_batch())
    names = {metric.name for metric in result.metrics}
    required = {
        "signed_residual_mean",
        "absolute_residual_mean",
        "shared_squared_error",
        "critic_one_loss",
        "critic_two_loss",
        "actor_loss",
        "actor_objective",
        "critic_one_gradient_norm",
        "critic_two_gradient_norm",
        "actor_gradient_norm",
        "q_abs_mean",
        "q_abs_max",
        "actor_shared_gradient_inner_product",
        "actor_shared_gradient_cosine",
    }
    required.update(
        f"{name}_{stat}" for name in ("y", "y1", "z1", "z") for stat in ("mean", "std", "abs_mean", "abs_max")
    )
    assert required <= names


@pytest.mark.parametrize("source", ["current", "next"])
def test_actor_state_source_matches_configured_input(source: str) -> None:
    algorithm = ThreeNetworkDDPG(_settings(actor_state_source=source))
    batch = _batch()
    states = batch.observations if source == "current" else batch.next_observations
    expected = -algorithm.critic_two(states, algorithm.actor(states)).mean()
    actual = algorithm._actor_loss(batch)
    torch.testing.assert_close(actual, expected)


def test_optional_diagnostic_does_not_change_training_update() -> None:
    batch = _batch()
    with_diagnostic = ThreeNetworkDDPG(_settings(diagnostic_frequency=1))
    without_diagnostic = ThreeNetworkDDPG(_settings(diagnostic_frequency=None))

    with_diagnostic.update_from_batch(batch)
    without_diagnostic.update_from_batch(batch)

    for name in ("actor", "critic_one", "critic_two"):
        with_state = getattr(with_diagnostic, name).state_dict()
        without_state = getattr(without_diagnostic, name).state_dict()
        for key in with_state:
            torch.testing.assert_close(with_state[key], without_state[key])


def test_configuration_and_checkpoint_round_trips(tmp_path: Path) -> None:
    config = load_config(
        algorithm="three_network_ddpg",
        experiment="three_network_core",
        overrides=[
            "algorithm.hidden_layers=[16,16]",
            "algorithm.update_schedule=simultaneous",
            "algorithm.actor_state_source=current",
            "algorithm.diagnostic_frequency=null",
        ],
    )
    settings = ThreeNetworkSettings.from_config(
        config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0], dtype=np.float32),
        action_high=np.array([2.0], dtype=np.float32),
    )
    assert ThreeNetworkSettings(**settings.to_dict()) == settings

    algorithm = ThreeNetworkDDPG(settings)
    algorithm.update_from_batch(_batch())
    checkpoint = tmp_path / "three-network.pt"
    algorithm.save_checkpoint(checkpoint)
    expected = deepcopy(algorithm.state_dict())

    restored = ThreeNetworkDDPG(settings)
    restored.restore_checkpoint(checkpoint)
    actual = restored.state_dict()
    for network in ("actor", "critic_one", "critic_two"):
        for key, value in expected[network].items():
            torch.testing.assert_close(value, actual[network][key])
    assert actual["settings"] == expected["settings"]
    assert actual["update_step"] == expected["update_step"]


def test_deterministic_evaluation_omits_collection_noise() -> None:
    algorithm = ThreeNetworkDDPG(_settings(exploration_noise_std=0.8))
    observation = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    first = algorithm.select_action(observation, deterministic=True)
    second = algorithm.select_action(observation, deterministic=True)
    np.testing.assert_array_equal(first.action, second.action)
    assert not first.exploration_applied
    exploratory = algorithm.select_action(observation, deterministic=False)
    assert exploratory.exploration_applied
    assert np.all(exploratory.action >= -2.0)
    assert np.all(exploratory.action <= 2.0)


@pytest.mark.parametrize("schedule", ["sequential", "simultaneous"])
def test_tiny_deterministic_environment_training_smoke_logs_return(schedule: str) -> None:
    config = load_config(
        algorithm="three_network_ddpg",
        environment="tiny_continuous",
        experiment="three_network_core",
        overrides=[
            "algorithm.hidden_layers=[8,8]",
            f"algorithm.update_schedule={schedule}",
            "algorithm.diagnostic_frequency=1",
            "experiment.total_steps=12",
            "experiment.warmup_steps=2",
            "experiment.batch_size=2",
            "experiment.replay_capacity=32",
            "experiment.evaluation_frequency=6",
            "experiment.evaluation_episodes=1",
        ],
    )
    environment = make_environment(config.environment)
    evaluation_environment = make_environment(config.environment)
    try:
        algorithm = ThreeNetworkDDPG.from_config(
            config.algorithm,
            observation_dim=1,
            action_low=environment.action_space.low,
            action_high=environment.action_space.high,
        )
        replay = ReplayBuffer(
            capacity=32,
            observation_shape=(1,),
            action_shape=(1,),
            seed=0,
            bootstrap_on_truncation=True,
            device="cpu",
        )
        summary = OffPolicyTrainer(
            algorithm=algorithm,
            environment=environment,
            replay=replay,
            settings=TrainerSettings(12, 2, 2, True, True),
        ).run(seed=0, deterministic_algorithms=True)
        evaluation = evaluate(
            algorithm,
            evaluation_environment,
            episodes=1,
            seed=100,
            clip_actions=True,
        )
        assert summary.environment_steps == 12
        assert summary.gradient_updates == 10
        assert "episode_return" in {metric.name for metric in summary.metrics}
        assert evaluation.lengths == (5,)
    finally:
        environment.close()
        evaluation_environment.close()


def test_short_pendulum_interaction_smoke() -> None:
    config = load_config(
        algorithm="three_network_ddpg",
        environment="pendulum_v1",
        experiment="three_network_core",
        overrides=[
            "algorithm.hidden_layers=[8,8]",
            "algorithm.diagnostic_frequency=null",
            "experiment.total_steps=12",
            "experiment.warmup_steps=4",
            "experiment.batch_size=4",
            "experiment.replay_capacity=32",
        ],
    )
    environment = make_environment(config.environment)
    try:
        algorithm = ThreeNetworkDDPG.from_config(
            config.algorithm,
            observation_dim=3,
            action_low=environment.action_space.low,
            action_high=environment.action_space.high,
        )
        replay = ReplayBuffer(
            capacity=32,
            observation_shape=(3,),
            action_shape=(1,),
            seed=0,
            bootstrap_on_truncation=True,
            device="cpu",
        )
        summary = OffPolicyTrainer(
            algorithm=algorithm,
            environment=environment,
            replay=replay,
            settings=TrainerSettings(12, 4, 4, True, True),
        ).run(seed=0, deterministic_algorithms=True)
        assert summary.environment_steps == 12
        assert summary.gradient_updates == 8
    finally:
        environment.close()


def test_three_network_checkpoint_normalizes_settings_and_allows_device_relocation() -> None:
    source = ThreeNetworkDDPG(_settings())
    state = deepcopy(source.state_dict())
    state["settings"]["hidden_layers"] = list(state["settings"]["hidden_layers"])
    state["settings"]["device"] = "cuda"

    restored = ThreeNetworkDDPG(_settings())
    restored.load_state_dict(state)

    for key, value in source.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[key])


def test_three_network_checkpoint_rejects_semantic_mismatch() -> None:
    state = deepcopy(ThreeNetworkDDPG(_settings()).state_dict())
    state["settings"]["gamma"] = 0.8

    with pytest.raises(ValueError, match="checkpoint settings do not match"):
        ThreeNetworkDDPG(_settings()).load_state_dict(state)
