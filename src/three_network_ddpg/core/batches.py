"""Transition, replay-batch, and on-policy rollout contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Transition:
    """One Gymnasium step, preserving both end-of-episode signals."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        for name in ("observation", "action", "next_observation"):
            array = np.asarray(getattr(self, name), dtype=np.float32)
            if array.ndim < 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite array with at least one dimension")
            object.__setattr__(self, name, array.copy())
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for name in ("terminated", "truncated"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Replay sample with leading batch dimension and an explicit bootstrap mask."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_mask: torch.Tensor

    def __post_init__(self) -> None:
        count = self.observations.shape[0]
        if self.observations.ndim < 2 or count == 0:
            raise ValueError("observations must have a nonempty leading batch dimension")
        if self.next_observations.shape != self.observations.shape:
            raise ValueError("next observations must match observation shape")
        if self.actions.ndim < 2 or self.actions.shape[0] != count:
            raise ValueError("actions must have the same batch length")
        if self.rewards.shape != (count, 1):
            raise ValueError("rewards must have shape (batch, 1)")
        for name in ("terminated", "truncated", "bootstrap_mask"):
            if getattr(self, name).shape != (count, 1):
                raise ValueError(f"{name} must have shape (batch, 1)")
        for name in ("observations", "actions", "rewards", "next_observations", "bootstrap_mask"):
            if getattr(self, name).dtype != torch.float32:
                raise TypeError(f"{name} must be float32")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must be boolean tensors")
        devices = {getattr(self, name).device for name in self.__dataclass_fields__}
        if len(devices) != 1:
            raise ValueError("all batch tensors must use the same device")

    @property
    def batch_size(self) -> int:
        return self.observations.shape[0]

    @property
    def device(self) -> torch.device:
        return self.observations.device

    def to(self, device: torch.device | str) -> "TransitionBatch":
        """Move every tensor without changing termination or truncation."""
        return TransitionBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """On-policy data; time and worker dimensions remain algorithm-owned."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = self.rewards.shape[0]
        if length == 0:
            raise ValueError("rollout must contain at least one step")
        for name in (
            "observations",
            "actions",
            "next_observations",
            "terminated",
            "truncated",
        ):
            if getattr(self, name).shape[0] != length:
                raise ValueError(f"{name} must match rollout length")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("rollout end flags must be boolean tensors")
