"""Opt-in shared-D reconstruction with explicitly delayed policy updates.

The critic equations are deliberately reused, not replaced by separate TD losses.
D_prime is retained only as a diagnostic. Delaying the actor is an experimental
schedule choice, not a claim about the advisor's exact implementation.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from ..core.batches import TransitionBatch
from ..core.interfaces import AlgorithmCapabilities, MetricRecord, UpdateResult
from .three_network_ddpg.implementation import (
    ThreeNetworkDDPG,
    ThreeNetworkSettings,
    _assign_gradients,
    _PhaseResult,
    agreement_error,
    shared_residual,
    shared_squared_error,
)


@dataclass(frozen=True, slots=True)
class SharedDSettings(ThreeNetworkSettings):
    """A distinct checkpoint/config identity; historical settings are unchanged."""

    agreement_mode: str = "both"
    alpha: float = 0.1
    alpha_prime: float = 0.1
    actor_update_delay: int = 2

    def __post_init__(self) -> None:
        ThreeNetworkSettings.__post_init__(self)
        if (
            isinstance(self.actor_update_delay, bool)
            or not isinstance(self.actor_update_delay, int)
            or self.actor_update_delay < 1
        ):
            raise ValueError("actor_update_delay must be a positive integer")
        if self.critic_objective != "shared_error" or self.agreement_mode != "both":
            raise ValueError("shared-D requires shared_error and both agreement penalties")
        if self.beta_mode != "none" or self.beta_coefficient != 0:
            raise ValueError("shared-D requires beta to be disabled")
        if self.update_schedule != "sequential" or self.actor_state_source != "next":
            raise ValueError("shared-D requires sequential updates and the next-state actor objective")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "SharedDSettings":
        # Reuse every existing validation/translation except the old delay=1 rule.
        core_config = dict(algorithm_config)
        core_config["actor_update_delay"] = 1
        base = ThreeNetworkSettings.from_config(
            core_config,
            observation_dim=observation_dim,
            action_low=action_low,
            action_high=action_high,
        )
        return cls(**base.to_dict(), actor_update_delay=algorithm_config["actor_update_delay"])


class SharedDDDPG(ThreeNetworkDDPG):
    """Q2 -> fresh Q1 -> Actor on every configured kth critic iteration."""

    capabilities = AlgorithmCapabilities("three_network_shared_d", False, True, True, "experimental")

    def __init__(self, settings: SharedDSettings) -> None:
        super().__init__(settings)
        self.settings: SharedDSettings = settings

    @classmethod
    def from_config(cls, algorithm_config: Mapping[str, Any], **kwargs: Any) -> "SharedDDDPG":
        return cls(SharedDSettings.from_config(algorithm_config, **kwargs))

    def _actor_update_due(self, next_step: int) -> bool:
        return next_step % self.settings.actor_update_delay == 0

    def _actor_update_count(self, step: int) -> int:
        return step // self.settings.actor_update_delay

    def _sequential_update(
        self, batch: TransitionBatch
    ) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult | None]:
        critic_two = self._sequential_critic_two(batch)
        critic_one = self._sequential_critic_one(batch)
        actor = None
        if self._actor_update_due(self._update_step + 1):
            actor = self._sequential_actor(batch)
        else:
            self._zero_all_gradients()
        return critic_one, critic_two, actor

    def update_from_batch(
        self, batch: TransitionBatch, *, validation_batch: TransitionBatch | None = None
    ) -> UpdateResult:
        result = super().update_from_batch(batch, validation_batch=validation_batch)
        step = self._update_step
        scheduling = (
            MetricRecord("actor_update_applied", float(self._actor_update_due(step)), step, "train"),
            MetricRecord("actor_update_steps", float(self._actor_update_count(step)), step, "train"),
        )
        return UpdateResult(
            metrics=result.metrics + scheduling,
            updates_applied=result.updates_applied,
            skipped_reason=result.skipped_reason,
        )


@dataclass(frozen=True, slots=True)
class BoardSharedDSettings(SharedDSettings):
    """Board reconstruction without the rejected Critic Two clock decimation."""

    actor_update_delay: int = 1
    actor_learning_starts: int = 0
    critic_initialization: Literal["independent", "identical"] = "independent"

    def __post_init__(self) -> None:
        SharedDSettings.__post_init__(self)
        if (
            isinstance(self.actor_learning_starts, bool)
            or not isinstance(self.actor_learning_starts, int)
            or self.actor_learning_starts < 0
        ):
            raise ValueError("board shared-D actor_learning_starts must be a nonnegative integer")
        if self.critic_initialization not in {"independent", "identical"}:
            raise ValueError("board shared-D critic_initialization must be independent or identical")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "BoardSharedDSettings":
        # The core parser intentionally supports only the main method's
        # independent initialization and one-step actor clock. This named
        # reconstruction owns the two explicit screening ablations.
        core_config = dict(algorithm_config)
        core_config["actor_update_delay"] = 1
        core_config["critic_initialization"] = "independent"
        base = ThreeNetworkSettings.from_config(
            core_config,
            observation_dim=observation_dim,
            action_low=action_low,
            action_high=action_high,
        )
        return cls(
            **base.to_dict(),
            actor_update_delay=int(algorithm_config["actor_update_delay"]),
            actor_learning_starts=int(algorithm_config["actor_learning_starts"]),
            critic_initialization=str(algorithm_config["critic_initialization"]),
        )


class BoardSharedDDDPG(SharedDDDPG):
    """Q2 -> recomputed Q1 -> Actor, with both critics updated every batch."""

    capabilities = AlgorithmCapabilities("three_network_shared_d_board", False, True, True, "experimental")

    def __init__(self, settings: BoardSharedDSettings) -> None:
        super().__init__(settings)
        self.settings: BoardSharedDSettings = settings
        if settings.critic_initialization == "identical":
            # Values start equal, but modules, parameters, and optimizers remain
            # independent immediately after construction.
            self.critic_two.load_state_dict(deepcopy(self.critic_one.state_dict()))

    @classmethod
    def from_config(cls, algorithm_config: Mapping[str, Any], **kwargs: Any) -> "BoardSharedDDDPG":
        return cls(BoardSharedDSettings.from_config(algorithm_config, **kwargs))

    def _actor_update_due(self, next_step: int) -> bool:
        learned_steps = next_step - self.settings.actor_learning_starts
        return learned_steps > 0 and learned_steps % self.settings.actor_update_delay == 0

    def _actor_update_count(self, step: int) -> int:
        learned_steps = max(0, step - self.settings.actor_learning_starts)
        return learned_steps // self.settings.actor_update_delay

    def update_from_batch(
        self, batch: TransitionBatch, *, validation_batch: TransitionBatch | None = None
    ) -> UpdateResult:
        result = super().update_from_batch(batch, validation_batch=validation_batch)
        step = self._update_step
        scheduling = (
            MetricRecord("critic_two_update_applied", 1.0, step, "train"),
            MetricRecord("critic_two_update_steps", float(step), step, "train"),
        )
        return UpdateResult(
            metrics=result.metrics + scheduling,
            updates_applied=result.updates_applied,
            skipped_reason=result.skipped_reason,
        )


@dataclass(frozen=True, slots=True)
class SynchronousSharedDSettings(ThreeNetworkSettings):
    """Shared-D critics computed from one snapshot, followed by a slower Actor."""

    actor_update_delay: int = 2
    actor_learning_starts: int = 5_000

    def __post_init__(self) -> None:
        ThreeNetworkSettings.__post_init__(self)
        if self.critic_objective != "shared_error" or self.agreement_mode != "both":
            raise ValueError("synchronous shared-D requires shared_error and both agreements")
        if self.beta_mode != "none" or self.beta_coefficient != 0:
            raise ValueError("synchronous shared-D requires beta to be disabled")
        if self.update_schedule != "simultaneous":
            raise ValueError("synchronous shared-D requires simultaneous critic gradients")
        if (
            isinstance(self.actor_learning_starts, bool)
            or not isinstance(self.actor_learning_starts, int)
            or self.actor_learning_starts < 0
        ):
            raise ValueError("synchronous shared-D actor_learning_starts must be nonnegative")
        if (
            isinstance(self.actor_update_delay, bool)
            or not isinstance(self.actor_update_delay, int)
            or self.actor_update_delay < 1
        ):
            raise ValueError("synchronous shared-D actor_update_delay must be positive")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "SynchronousSharedDSettings":
        core_config = dict(algorithm_config)
        core_config["actor_update_delay"] = 1
        base = ThreeNetworkSettings.from_config(
            core_config,
            observation_dim=observation_dim,
            action_low=action_low,
            action_high=action_high,
        )
        return cls(
            **base.to_dict(),
            actor_update_delay=int(algorithm_config["actor_update_delay"]),
            actor_learning_starts=int(algorithm_config["actor_learning_starts"]),
        )


class SynchronousSharedDDDPG(ThreeNetworkDDPG):
    """Same-snapshot Q2/Q1 gradients, commit both critics, then update the Actor."""

    capabilities = AlgorithmCapabilities(
        "three_network_shared_d_synchronous", False, True, True, "experimental"
    )

    def __init__(self, settings: SynchronousSharedDSettings) -> None:
        super().__init__(settings)
        self.settings: SynchronousSharedDSettings = settings

    @classmethod
    def from_config(cls, algorithm_config: Mapping[str, Any], **kwargs: Any) -> "SynchronousSharedDDDPG":
        return cls(SynchronousSharedDSettings.from_config(algorithm_config, **kwargs))

    def _actor_update_due(self, next_step: int) -> bool:
        learned_steps = next_step - self.settings.actor_learning_starts
        return learned_steps > 0 and learned_steps % self.settings.actor_update_delay == 0

    def _actor_update_count(self, step: int) -> int:
        learned_steps = max(0, step - self.settings.actor_learning_starts)
        return learned_steps // self.settings.actor_update_delay

    def _simultaneous_update(
        self, batch: TransitionBatch
    ) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult | None]:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)
        y = self.critic_one(batch.observations, batch.actions)
        y_next = self.critic_one(batch.next_observations, next_actions)
        z_current = self.critic_two(batch.observations, batch.actions)
        z_next = self.critic_two(batch.next_observations, next_actions)

        critic_two_shared = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z_next,
                y.detach(),
                self.settings.gamma,
            )
        )
        current_agreement = agreement_error(z_current, y.detach())
        critic_two_loss = critic_two_shared + self.settings.alpha_prime * current_agreement
        critic_two_parameters = tuple(self.critic_two.parameters())
        critic_two_gradients = torch.autograd.grad(critic_two_loss, critic_two_parameters)

        critic_one_shared = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z_next.detach(),
                y,
                self.settings.gamma,
            )
        )
        next_agreement = agreement_error(y_next, z_next.detach())
        critic_one_loss = critic_one_shared + self.settings.alpha * next_agreement
        critic_one_parameters = tuple(self.critic_one.parameters())
        critic_one_gradients = torch.autograd.grad(critic_one_loss, critic_one_parameters)

        zero = critic_one_shared.new_zeros(())
        critic_one_result = self._phase_result(
            critic_one_loss,
            critic_one_shared,
            next_agreement,
            zero,
            critic_one_gradients,
        )
        critic_two_result = self._phase_result(
            critic_two_loss,
            critic_two_shared,
            current_agreement,
            zero,
            critic_two_gradients,
        )

        # Both gradient sets come from the same pre-update critic snapshot. The
        # optimizer commits happen only after both sets have been calculated.
        _assign_gradients(critic_two_parameters, critic_two_gradients)
        _assign_gradients(critic_one_parameters, critic_one_gradients)
        self._clip_assigned_gradients(critic_two_parameters)
        self._clip_assigned_gradients(critic_one_parameters)
        self.critic_two_optimizer.step()
        self.critic_one_optimizer.step()

        actor = None
        if self._actor_update_due(self._update_step + 1):
            actor = self._sequential_actor(batch)
        else:
            self._zero_all_gradients()
        return critic_one_result, critic_two_result, actor

    def update_from_batch(
        self, batch: TransitionBatch, *, validation_batch: TransitionBatch | None = None
    ) -> UpdateResult:
        result = super().update_from_batch(batch, validation_batch=validation_batch)
        step = self._update_step
        scheduling = (
            MetricRecord("critic_snapshot_synchronous", 1.0, step, "train"),
            MetricRecord("critic_two_update_applied", 1.0, step, "train"),
            MetricRecord("critic_two_update_steps", float(step), step, "train"),
            MetricRecord("actor_update_applied", float(self._actor_update_due(step)), step, "train"),
            MetricRecord("actor_update_steps", float(self._actor_update_count(step)), step, "train"),
        )
        return UpdateResult(
            metrics=result.metrics + scheduling,
            updates_applied=result.updates_applied,
            skipped_reason=result.skipped_reason,
        )


@dataclass(frozen=True, slots=True)
class ConditionedSharedDSettings(SharedDSettings):
    """Explicit asymmetric coupling for the ill-conditioned shared-D system."""

    alpha: float = 0.01
    alpha_prime: float = 10.0
    actor_learning_starts: int = 5_000

    def __post_init__(self) -> None:
        SharedDSettings.__post_init__(self)
        if (
            isinstance(self.actor_learning_starts, bool)
            or not isinstance(self.actor_learning_starts, int)
            or self.actor_learning_starts < 0
        ):
            raise ValueError("actor_learning_starts must be a nonnegative integer")
        if self.alpha <= 0 or self.alpha_prime <= self.gamma * self.alpha:
            raise ValueError("conditioned shared-D requires alpha_prime > gamma * alpha > 0")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "ConditionedSharedDSettings":
        core_config = dict(algorithm_config)
        core_config["actor_update_delay"] = 1
        base = ThreeNetworkSettings.from_config(
            core_config,
            observation_dim=observation_dim,
            action_low=action_low,
            action_high=action_high,
        )
        return cls(
            **base.to_dict(),
            actor_update_delay=int(algorithm_config["actor_update_delay"]),
            actor_learning_starts=int(algorithm_config["actor_learning_starts"]),
        )


class ConditionedSharedDDDPG(SharedDDDPG):
    """Shared-D with asymmetric agreement and a critic-only preparation phase."""

    capabilities = AlgorithmCapabilities(
        "three_network_shared_d_conditioned", False, True, True, "experimental"
    )

    def __init__(self, settings: ConditionedSharedDSettings) -> None:
        super().__init__(settings)
        self.settings: ConditionedSharedDSettings = settings

    @classmethod
    def from_config(cls, algorithm_config: Mapping[str, Any], **kwargs: Any) -> "ConditionedSharedDDDPG":
        return cls(ConditionedSharedDSettings.from_config(algorithm_config, **kwargs))

    def _actor_update_due(self, next_step: int) -> bool:
        learned_steps = next_step - self.settings.actor_learning_starts
        return learned_steps > 0 and learned_steps % self.settings.actor_update_delay == 0

    def _actor_update_count(self, step: int) -> int:
        learned_steps = max(0, step - self.settings.actor_learning_starts)
        return learned_steps // self.settings.actor_update_delay


@dataclass(frozen=True, slots=True)
class TwoTimescaleSharedDSettings(SharedDSettings):
    """Shared-D with explicit Q1-fast, Q2-slow, Actor-slowest update clocks."""

    alpha: float = 0.1
    alpha_prime: float = 0.1
    actor_learning_starts: int = 5_000
    critic_two_update_delay: int = 2

    def __post_init__(self) -> None:
        SharedDSettings.__post_init__(self)
        for name, value, minimum in (
            ("actor_learning_starts", self.actor_learning_starts, 0),
            ("critic_two_update_delay", self.critic_two_update_delay, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.critic_two_learning_rate >= self.critic_one_learning_rate:
            raise ValueError("two-timescale shared-D requires Critic Two to learn slower than Critic One")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "TwoTimescaleSharedDSettings":
        core_config = dict(algorithm_config)
        core_config["actor_update_delay"] = 1
        base = ThreeNetworkSettings.from_config(
            core_config,
            observation_dim=observation_dim,
            action_low=action_low,
            action_high=action_high,
        )
        return cls(
            **base.to_dict(),
            actor_update_delay=int(algorithm_config["actor_update_delay"]),
            actor_learning_starts=int(algorithm_config["actor_learning_starts"]),
            critic_two_update_delay=int(algorithm_config["critic_two_update_delay"]),
        )


class TwoTimescaleSharedDDDPG(SharedDDDPG):
    """Use one board-defined minibatch while separating the three update clocks."""

    capabilities = AlgorithmCapabilities(
        "three_network_shared_d_two_timescale", False, True, True, "experimental"
    )

    def __init__(self, settings: TwoTimescaleSharedDSettings) -> None:
        super().__init__(settings)
        self.settings: TwoTimescaleSharedDSettings = settings

    @classmethod
    def from_config(cls, algorithm_config: Mapping[str, Any], **kwargs: Any) -> "TwoTimescaleSharedDDDPG":
        return cls(TwoTimescaleSharedDSettings.from_config(algorithm_config, **kwargs))

    def _critic_two_update_due(self, next_step: int) -> bool:
        return (next_step - 1) % self.settings.critic_two_update_delay == 0

    def _critic_two_update_count(self, step: int) -> int:
        if step < 1:
            return 0
        return (step - 1) // self.settings.critic_two_update_delay + 1

    def _eligible_critic_two_updates(self, step: int) -> int:
        completed = self._critic_two_update_count(step)
        before_actor = self._critic_two_update_count(self.settings.actor_learning_starts)
        return max(0, completed - before_actor)

    def _actor_update_due(self, next_step: int) -> bool:
        if not self._critic_two_update_due(next_step):
            return False
        eligible = self._eligible_critic_two_updates(next_step)
        return eligible > 0 and eligible % self.settings.actor_update_delay == 0

    def _actor_update_count(self, step: int) -> int:
        return self._eligible_critic_two_updates(step) // self.settings.actor_update_delay

    def _sequential_update(
        self, batch: TransitionBatch
    ) -> tuple[_PhaseResult, _PhaseResult | None, _PhaseResult | None]:
        # Every phase in this learner step uses the same transition minibatch B.
        # Only its board-defined views differ: (s, a) for Y/Z1 and
        # (s_next, actor(s_next)) for Y1/Z.
        critic_two = None
        if self._critic_two_update_due(self._update_step + 1):
            critic_two = self._sequential_critic_two(batch)
        critic_one = self._sequential_critic_one(batch)
        actor = None
        if self._actor_update_due(self._update_step + 1):
            actor = self._sequential_actor(batch)
        else:
            self._zero_all_gradients()
        return critic_one, critic_two, actor

    def update_from_batch(
        self, batch: TransitionBatch, *, validation_batch: TransitionBatch | None = None
    ) -> UpdateResult:
        result = super().update_from_batch(batch, validation_batch=validation_batch)
        step = self._update_step
        scheduling = (
            MetricRecord(
                "critic_two_update_applied", float(self._critic_two_update_due(step)), step, "train"
            ),
            MetricRecord(
                "critic_two_update_steps", float(self._critic_two_update_count(step)), step, "train"
            ),
        )
        return UpdateResult(
            metrics=result.metrics + scheduling,
            updates_applied=result.updates_applied,
            skipped_reason=result.skipped_reason,
        )
