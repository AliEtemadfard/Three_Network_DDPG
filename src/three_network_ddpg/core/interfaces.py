"""Capability-aware algorithm contracts without shared loss logic."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .batches import RolloutBatch, Transition, TransitionBatch

ImplementationStatus = Literal["planned", "experimental", "verified"]
MetricPhase = Literal["train", "evaluation"]


@dataclass(frozen=True, slots=True)
class AlgorithmCapabilities:
    name: str
    is_on_policy: bool
    uses_replay_buffer: bool
    supports_continuous_actions: bool
    implementation_status: ImplementationStatus

    def __post_init__(self) -> None:
        if self.is_on_policy and self.uses_replay_buffer:
            raise ValueError("on-policy algorithms cannot use the off-policy replay buffer")


@dataclass(frozen=True, slots=True)
class ActionSelection:
    """Environment-scale action and optional scalar policy diagnostics."""

    action: np.ndarray
    exploration_applied: bool
    auxiliary: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32)
        if action.ndim < 1 or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array with at least one dimension")
        if not isinstance(self.exploration_applied, bool):
            raise TypeError("exploration_applied must be boolean")
        object.__setattr__(self, "action", action.copy())


@dataclass(frozen=True, slots=True)
class MetricRecord:
    name: str
    value: float
    step: int
    phase: MetricPhase

    def __post_init__(self) -> None:
        if not self.name or self.step < 0 or not np.isfinite(self.value):
            raise ValueError("metric name, step, and finite value are required")


@dataclass(frozen=True, slots=True)
class UpdateResult:
    metrics: tuple[MetricRecord, ...] = ()
    updates_applied: int = 0
    skipped_reason: str | None = None

    def __post_init__(self) -> None:
        if self.updates_applied < 0:
            raise ValueError("updates_applied cannot be negative")
        if self.updates_applied and self.skipped_reason is not None:
            raise ValueError("an applied update cannot also be skipped")


class Algorithm(ABC):
    """Shared policy and state contract for off-policy and on-policy methods."""

    capabilities: AlgorithmCapabilities

    @abstractmethod
    def select_action(self, observation: np.ndarray, *, deterministic: bool) -> ActionSelection:
        """Return an environment-scale action; evaluation must omit exploration."""

    @abstractmethod
    def set_evaluation_mode(self, enabled: bool) -> None:
        """Switch policy and network behavior for evaluation or training."""

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return serializable model, optimizer, and schedule state."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore model, optimizer, and schedule state."""

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save an explicit checkpoint at the given path."""

    @abstractmethod
    def restore_checkpoint(self, path: Path) -> None:
        """Restore an explicit checkpoint from the given path."""


class OffPolicyAlgorithm(Algorithm):
    """Only this interface participates in a replay-driven trainer."""

    @abstractmethod
    def observe_transition(self, transition: Transition) -> None:
        """Receive the unmodified transition after environment interaction."""

    @abstractmethod
    def update_from_batch(self, batch: TransitionBatch) -> UpdateResult:
        """Apply algorithm-owned updates using an off-policy replay sample."""


class OnPolicyAlgorithm(Algorithm):
    """Separate contract for rollouts; no replay-loop inheritance."""

    @abstractmethod
    def observe_rollout(self, rollout: RolloutBatch) -> None:
        """Receive a complete algorithm-defined rollout."""

    @abstractmethod
    def update_from_rollout(self, rollout: RolloutBatch) -> UpdateResult:
        """Apply algorithm-owned on-policy updates."""
