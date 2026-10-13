"""Opt-in advisor reconstruction with aligned replay rows and a slower Actor.

The objectives remain shared-D, not D-prime regression. One critic iteration
updates both critics from the same transition rows, then optionally the Actor.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..core.batches import TransitionBatch
from ..core.interfaces import AlgorithmCapabilities, MetricRecord, UpdateResult
from .shared_d import BoardSharedDDDPG, BoardSharedDSettings
from .three_network_ddpg.implementation import (
    _assign_gradients,
    _PhaseResult,
    agreement_error,
    shared_residual,
    shared_squared_error,
)


@dataclass(frozen=True, slots=True)
class PairedSharedDSettings(BoardSharedDSettings):
    """A separate semantic/checkpoint identity; no historical defaults change."""

    actor_update_delay: int = 10
    actor_learning_starts: int = 0
    paired_batch_contract: Literal["board_aligned_v1"] = "board_aligned_v1"

    def __post_init__(self) -> None:
        BoardSharedDSettings.__post_init__(self)
        if self.critic_initialization != "independent":
            raise ValueError("paired shared-D requires independent critic initialization")
        if self.actor_update_delay < 2:
            raise ValueError("paired shared-D requires more critic-pair updates than Actor updates")
        if self.paired_batch_contract != "board_aligned_v1":
            raise ValueError("paired shared-D requires paired_batch_contract=board_aligned_v1")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> "PairedSharedDSettings":
        base = BoardSharedDSettings.from_config(algorithm_config, **kwargs)
        return cls(
            **base.to_dict(),
            paired_batch_contract=algorithm_config["paired_batch_contract"],
        )


@dataclass(frozen=True, slots=True)
class _PairedInputs:
    """One replay ordering and one detached next-action vector for both critics."""

    batch: TransitionBatch
    next_actions: torch.Tensor


class PairedSharedDDDPG(BoardSharedDDDPG):
    """Q2 -> fresh Q2 outputs for Q1 -> Q1 -> delayed next-state Actor."""

    capabilities = AlgorithmCapabilities("three_network_shared_d_paired", False, True, True, "experimental")

    def __init__(self, settings: PairedSharedDSettings) -> None:
        super().__init__(settings)
        self.settings: PairedSharedDSettings = settings

    @classmethod
    def from_config(cls, algorithm_config: Mapping[str, Any], **kwargs: Any) -> "PairedSharedDDDPG":
        return cls(PairedSharedDSettings.from_config(algorithm_config, **kwargs))

    def _paired_inputs(self, batch: TransitionBatch) -> _PairedInputs:
        # All four board vectors use row i of this single TransitionBatch.
        # Never sample, permute, or independently pair current/next columns here.
        return _PairedInputs(batch, self._next_actions_detached(batch))

    def _paired_critic_two(self, inputs: _PairedInputs) -> _PhaseResult:
        self._zero_all_gradients()
        batch, a_next = inputs.batch, inputs.next_actions
        with torch.no_grad():
            y = self.critic_one(batch.observations, batch.actions)
        z_current = self.critic_two(batch.observations, batch.actions)
        z_next = self.critic_two(batch.next_observations, a_next)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z_next, y, self.settings.gamma)
        )
        current = agreement_error(z_current, y)
        total = shared + self.settings.alpha_prime * current
        # D-prime is diagnostic only. Its detached target does not replace D.
        target = (batch.rewards + self.settings.gamma * batch.bootstrap_mask * z_next).detach()
        td_diagnostic = shared_squared_error(z_current.detach() - target)
        parameters = tuple(self.critic_two.parameters())
        gradients = torch.autograd.grad(total, parameters)
        result = self._phase_result(total, shared, current, shared.new_zeros(()), gradients, td=td_diagnostic)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.critic_two_optimizer.step()
        return result

    def _paired_critic_one(self, inputs: _PairedInputs) -> _PhaseResult:
        self._zero_all_gradients()
        batch, a_next = inputs.batch, inputs.next_actions
        # Same next inputs, but fresh Q2 VALUES after its optimizer step.
        with torch.no_grad():
            z_next = self.critic_two(batch.next_observations, a_next)
        y = self.critic_one(batch.observations, batch.actions)
        y_next = self.critic_one(batch.next_observations, a_next)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z_next, y, self.settings.gamma)
        )
        following = agreement_error(y_next, z_next)
        total = shared + self.settings.alpha * following
        parameters = tuple(self.critic_one.parameters())
        gradients = torch.autograd.grad(total, parameters)
        result = self._phase_result(total, shared, following, shared.new_zeros(()), gradients)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.critic_one_optimizer.step()
        return result

    def _sequential_update(
        self, batch: TransitionBatch
    ) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult | None]:
        inputs = self._paired_inputs(batch)
        critic_two = self._paired_critic_two(inputs)
        critic_one = self._paired_critic_one(inputs)
        actor = None
        if self._actor_update_due(self._update_step + 1):
            # Fresh attached Actor actions on the SAME ordered s_next rows.
            actor = self._sequential_actor(batch)
        else:
            self._zero_all_gradients()
        return critic_one, critic_two, actor

    def update_from_batch(
        self, batch: TransitionBatch, *, validation_batch: TransitionBatch | None = None
    ) -> UpdateResult:
        result = super().update_from_batch(batch, validation_batch=validation_batch)
        return UpdateResult(
            metrics=result.metrics
            + (
                MetricRecord(
                    "critic_pair_update_steps", float(self._update_step), self._update_step, "train"
                ),
                MetricRecord("paired_batch_size", float(batch.batch_size), self._update_step, "train"),
            ),
            updates_applied=result.updates_applied,
            skipped_reason=result.skipped_reason,
        )


@dataclass(frozen=True, slots=True)
class ConditionedPairedSharedDSettings(PairedSharedDSettings):
    """Board-aligned inputs with the previously useful current-value anchor."""

    alpha: float = 0.01
    alpha_prime: float = 10.0
    actor_update_delay: int = 10
    actor_learning_starts: int = 5_000
    conditioning_contract: Literal["asymmetric_current_anchor_v1"] = "asymmetric_current_anchor_v1"

    def __post_init__(self) -> None:
        PairedSharedDSettings.__post_init__(self)
        if self.alpha != 0.01 or self.alpha_prime != 10.0:
            raise ValueError("conditioned paired shared-D requires alpha=0.01 and alpha_prime=10.0")
        if self.actor_learning_starts != 5_000:
            raise ValueError("conditioned paired shared-D requires actor_learning_starts=5000")
        if self.conditioning_contract != "asymmetric_current_anchor_v1":
            raise ValueError(
                "conditioned paired shared-D requires conditioning_contract=asymmetric_current_anchor_v1"
            )

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> "ConditionedPairedSharedDSettings":
        base = PairedSharedDSettings.from_config(algorithm_config, **kwargs)
        return cls(
            **base.to_dict(),
            conditioning_contract=algorithm_config["conditioning_contract"],
        )


class ConditionedPairedSharedDDDPG(PairedSharedDDDPG):
    """Paired shared-D with an explicit asymmetric Q2-current anchor."""

    capabilities = AlgorithmCapabilities(
        "three_network_shared_d_conditioned_paired", False, True, True, "experimental"
    )

    def __init__(self, settings: ConditionedPairedSharedDSettings) -> None:
        super().__init__(settings)
        self.settings: ConditionedPairedSharedDSettings = settings

    @classmethod
    def from_config(
        cls, algorithm_config: Mapping[str, Any], **kwargs: Any
    ) -> "ConditionedPairedSharedDDDPG":
        return cls(ConditionedPairedSharedDSettings.from_config(algorithm_config, **kwargs))
