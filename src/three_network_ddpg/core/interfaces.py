"""A small capability-based interface, without a shared training loop."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .batches import RolloutBatch, TransitionBatch

ImplementationStatus = Literal["planned", "experimental", "verified"]


@dataclass(frozen=True)
class AlgorithmCapabilities:
    name: str
    is_on_policy: bool
    uses_replay_buffer: bool
    supports_continuous_actions: bool
    implementation_status: ImplementationStatus


@dataclass(frozen=True)
class UpdateMetrics:
    """Structured scalar metrics from one future update."""

    scalars: Mapping[str, float]
    environment_steps: int = 0
    gradient_steps: int = 0


class Algorithm(ABC):
    """Contract for algorithms with either replay transitions or rollouts."""

    capabilities: AlgorithmCapabilities

    @abstractmethod
    def select_action(self, observation: Any, *, deterministic: bool) -> Any:
        """Return an action; deterministic evaluation must omit exploration noise."""

    @abstractmethod
    def observe(self, data: TransitionBatch[Any] | RolloutBatch[Any]) -> None:
        """Accept data appropriate to this algorithm's capabilities."""

    @abstractmethod
    def update(self) -> UpdateMetrics:
        """Execute one algorithm-defined update, if sufficient data exists."""

    @abstractmethod
    def set_evaluation_mode(self, enabled: bool) -> None:
        """Switch policy and network behavior for evaluation or training."""

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return serializable model, optimizer, and schedule state."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state, including optimizer and schedule counters."""

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save an explicit checkpoint at the given path."""

    @abstractmethod
    def restore_checkpoint(self, path: Path) -> None:
        """Restore an explicit checkpoint from the given path."""
