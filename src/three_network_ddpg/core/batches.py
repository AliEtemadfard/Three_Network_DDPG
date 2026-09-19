"""Typed containers; tensor libraries remain an implementation choice."""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TransitionBatch(Generic[T]):
    """Off-policy transition data with distinct termination and truncation flags."""

    observations: T
    actions: T
    rewards: T
    next_observations: T
    terminated: T
    truncated: T


@dataclass(frozen=True)
class RolloutBatch(Generic[T]):
    """On-policy rollout data; algorithms may attach their own auxiliary state."""

    observations: T
    actions: T
    rewards: T
    next_observations: T
    terminated: T
    truncated: T
