"""Only the explicitly configured continuous-control environments are constructed."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TINY_ENV_ID = "ThreeNetworkTinyContinuous-v0"


class TinyContinuousEnv(gym.Env[np.ndarray, np.ndarray]):
    """One-dimensional deterministic dynamics for contract and smoke tests."""

    metadata = {"render_modes": []}

    def __init__(self, *, horizon: int) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        observation_bound = np.float32(0.2 * horizon)
        self.observation_space = spaces.Box(
            low=np.array([-observation_bound], dtype=np.float32),
            high=np.array([observation_bound], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._state: np.float32 | None = None
        self._steps = 0
        self._finished = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = np.float32(0.0)
        self._steps = 0
        self._finished = False
        return np.array([self._state], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None or self._finished:
            raise RuntimeError("reset before stepping a new episode")
        candidate = np.asarray(action, dtype=np.float32)
        if candidate.shape != self.action_space.shape or not self.action_space.contains(candidate):
            raise ValueError("action must be a finite in-bounds Box action")
        self._state = np.float32(self._state + np.float32(0.1) * candidate[0])
        self._steps += 1
        terminated = False
        truncated = self._steps >= self.horizon
        self._finished = truncated
        reward = -float((self._state - np.float32(1.0)) ** 2)
        return np.array([self._state], dtype=np.float32), reward, terminated, truncated, {}


def require_continuous_spaces(environment: gym.Env) -> None:
    """Reject non-Box spaces and unbounded action coordinates."""
    if not isinstance(environment.observation_space, spaces.Box):
        raise TypeError("continuous-control harness requires a Box observation space")
    if not isinstance(environment.action_space, spaces.Box):
        raise TypeError("continuous-control harness requires a Box action space")
    if not np.all(np.isfinite(environment.action_space.low)) or not np.all(
        np.isfinite(environment.action_space.high)
    ):
        raise ValueError("continuous actions require finite bounds")


def make_environment(config: Mapping[str, Any]) -> gym.Env:
    """Construct a configured tiny fixture or Pendulum-v1 without hidden wrappers."""
    environment_id = config["id"]
    wrappers = config["wrappers"]
    if wrappers:
        raise NotImplementedError("environment wrappers are not implemented in this core")
    if environment_id == TINY_ENV_ID:
        environment: gym.Env = TinyContinuousEnv(horizon=config["horizon"])
    elif environment_id == "Pendulum-v1":
        environment = gym.make("Pendulum-v1")
    else:
        raise ValueError(f"unsupported environment id: {environment_id}")
    require_continuous_spaces(environment)
    return environment
