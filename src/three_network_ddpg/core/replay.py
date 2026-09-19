"""CPU-backed replay storage with explicit terminal and truncation policy."""

import numpy as np
import torch

from .batches import Transition, TransitionBatch


def bootstrap_mask(
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    bootstrap_on_truncation: bool,
) -> torch.Tensor:
    """Return 0 for terminals and optionally for time-limit truncations."""
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean tensors")
    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated shapes must match")
    valid = ~terminated
    if not bootstrap_on_truncation:
        valid = valid & ~truncated
    return valid.to(dtype=torch.float32)


class ReplayBuffer:
    """Fixed-capacity transition ring; sampling returns detached tensor copies."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        seed: int,
        bootstrap_on_truncation: bool,
        device: torch.device | str,
    ) -> None:
        if capacity <= 0 or not observation_shape or not action_shape:
            raise ValueError("capacity and nonempty observation/action shapes are required")
        if any(size <= 0 for size in (*observation_shape, *action_shape)):
            raise ValueError("observation and action dimensions must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if not isinstance(bootstrap_on_truncation, bool):
            raise TypeError("bootstrap_on_truncation must be boolean")
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.device = requested_device
        self._rng = np.random.default_rng(seed)
        self._observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self._actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self._terminated = np.empty((capacity, 1), dtype=np.bool_)
        self._truncated = np.empty((capacity, 1), dtype=np.bool_)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        """Copy an environment transition without merging end flags."""
        if transition.observation.shape != self.observation_shape:
            raise ValueError("observation shape does not match replay buffer")
        if transition.next_observation.shape != self.observation_shape:
            raise ValueError("next observation shape does not match replay buffer")
        if transition.action.shape != self.action_shape:
            raise ValueError("action shape does not match replay buffer")
        index = self._position
        self._observations[index] = transition.observation
        self._actions[index] = transition.action
        self._rewards[index, 0] = transition.reward
        self._next_observations[index] = transition.next_observation
        self._terminated[index, 0] = transition.terminated
        self._truncated[index, 0] = transition.truncated
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, device: torch.device | str | None = None) -> TransitionBatch:
        """Sample without replacement and move every field to the selected device."""
        if batch_size <= 0 or batch_size > self._size:
            raise ValueError("batch_size must be in [1, number of stored transitions]")
        target = self.device if device is None else torch.device(device)
        indices = self._rng.choice(self._size, size=batch_size, replace=False)

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array[indices].copy(), device=target)

        terminated = tensor(self._terminated)
        truncated = tensor(self._truncated)
        return TransitionBatch(
            observations=tensor(self._observations),
            actions=tensor(self._actions),
            rewards=tensor(self._rewards),
            next_observations=tensor(self._next_observations),
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask(
                terminated, truncated, bootstrap_on_truncation=self.bootstrap_on_truncation
            ),
        )
