"""Deterministic seed behavior and small environment interactions."""

import random

import numpy as np
import pytest
import torch

from three_network_ddpg.cli import main
from three_network_ddpg.config import load_config
from three_network_ddpg.core.seeds import seed_environment, seed_everything
from three_network_ddpg.environments import TinyContinuousEnv, make_environment


def test_seed_repeats_python_numpy_and_torch() -> None:
    first_report = seed_everything(23, deterministic_algorithms=True)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    second_report = seed_everything(23, deterministic_algorithms=True)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second
    assert first_report == second_report
    assert first_report.python_seeded
    assert first_report.numpy_seeded
    assert first_report.torch_seeded
    assert first_report.cuda_available is torch.cuda.is_available()
    assert first_report.deterministic_algorithms_requested
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(23, deterministic_algorithms=False)
    with pytest.raises(ValueError, match="seed must"):
        seed_everything(2**32)


def test_gymnasium_action_space_and_reset_are_seeded() -> None:
    left = TinyContinuousEnv(horizon=3)
    right = TinyContinuousEnv(horizon=3)
    try:
        left_observation, _ = seed_environment(left, 11)
        right_observation, _ = seed_environment(right, 11)
        np.testing.assert_array_equal(left_observation, right_observation)
        np.testing.assert_array_equal(left.action_space.sample(), right.action_space.sample())
    finally:
        left.close()
        right.close()


def test_tiny_horizon_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TinyContinuousEnv(horizon=1.5)


def test_tiny_dynamics_and_truncation() -> None:
    environment = make_environment(load_config(environment="tiny_continuous").environment)
    try:
        seed_environment(environment, 4)
        for index in range(5):
            observation, reward, terminated, truncated, _ = environment.step(
                np.array([1.0], dtype=np.float32)
            )
            np.testing.assert_allclose(observation, [(index + 1) * 0.1], atol=1e-6)
            assert reward <= 0
            assert not terminated
            assert truncated is (index == 4)
        with pytest.raises(RuntimeError, match="reset"):
            environment.step(np.array([0.0], dtype=np.float32))
    finally:
        environment.close()


def test_pendulum_random_policy_short_smoke() -> None:
    environment = make_environment(load_config(environment="pendulum_v1").environment)
    try:
        observation, _ = seed_environment(environment, 6)
        assert observation.shape == (3,)
        for _ in range(3):
            action = environment.action_space.sample()
            observation, reward, terminated, truncated, _ = environment.step(action)
            assert observation.shape == (3,)
            assert np.isfinite(reward)
            assert not terminated and not truncated
    finally:
        environment.close()


def test_cli_random_policy_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env-smoke", "--environment", "tiny_continuous", "--steps", "6", "--seed", "2"]) == 0
    output = capsys.readouterr().out
    assert "steps=6" in output
    assert "truncated=1" in output
