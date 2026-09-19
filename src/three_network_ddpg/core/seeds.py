"""Seed Python, NumPy, PyTorch, and Gymnasium where supported."""

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class SeedReport:
    seed: int
    deterministic_algorithms_requested: bool
    cuda_available: bool


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_NUMPY_SEED}]")


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> SeedReport:
    """Seed global RNGs; exact determinism across devices is not guaranteed."""
    _check_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic_algorithms
        torch.backends.cudnn.benchmark = not deterministic_algorithms
    return SeedReport(seed, deterministic_algorithms, cuda_available)


def seed_action_space(action_space: Any, seed: int) -> None:
    """Seed Gymnasium action sampling independently of the environment RNG."""
    _check_seed(seed)
    action_space.seed(seed)


def seed_environment(environment: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    """Seed action space and reset the environment; retain Gymnasium's info."""
    _check_seed(seed)
    seed_action_space(environment.action_space, seed)
    if hasattr(environment, "observation_space"):
        environment.observation_space.seed(seed)
    observation, info = environment.reset(seed=seed)
    return observation, info
