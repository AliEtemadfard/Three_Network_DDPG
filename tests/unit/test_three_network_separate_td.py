"""Focused contracts for the opt-in separate-TD reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from three_network_ddpg.algorithms.three_network_ddpg import (
    ThreeNetworkDDPG,
    ThreeNetworkSettings,
    q2_td_residual,
    shared_residual,
    shared_squared_error,
)
from three_network_ddpg.config import load_config
from three_network_ddpg.core.batches import TransitionBatch
from three_network_ddpg.protocol.manifest import expand_manifest, load_manifest

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY / "experiments" / "manifests" / "pendulum_separate_td_screening_01.yaml"


def _settings(**changes: Any) -> ThreeNetworkSettings:
    values: dict[str, Any] = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
        "gamma": 0.95,
        "actor_learning_rate": 1e-3,
        "critic_one_learning_rate": 1e-3,
        "critic_two_learning_rate": 1e-3,
        "critic_objective": "separate_td",
        "agreement_mode": "both",
        "alpha": 0.3,
        "alpha_prime": 0.4,
        "beta_mode": "none",
        "beta_coefficient": 0.0,
        "update_schedule": "sequential",
        "actor_state_source": "next",
        "diagnostic_frequency": None,
        "seed": 47,
    }
    values.update(changes)
    return ThreeNetworkSettings(**values)


def _batch(seed: int = 9, batch_size: int = 7) -> TransitionBatch:
    generator = torch.Generator().manual_seed(seed)
    flags = torch.zeros((batch_size, 1), dtype=torch.bool)
    mask = torch.ones((batch_size, 1))
    mask[-1] = 0.0
    return TransitionBatch(
        observations=torch.randn((batch_size, 3), generator=generator),
        actions=torch.empty((batch_size, 1)).uniform_(-2.0, 2.0, generator=generator),
        rewards=torch.randn((batch_size, 1), generator=generator),
        next_observations=torch.randn((batch_size, 3), generator=generator),
        terminated=flags,
        truncated=flags.clone(),
        bootstrap_mask=mask,
    )


def _copies(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _assert_unchanged(before: tuple[torch.Tensor, ...], module: torch.nn.Module) -> None:
    for expected, actual in zip(before, module.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_gradients_match(expected: tuple[torch.Tensor, ...], module: torch.nn.Module) -> None:
    for reference, parameter in zip(expected, module.parameters(), strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference)


def test_q2_separate_td_loss_and_current_input_gradient_match_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(_settings(agreement_mode="none", alpha=0.0, alpha_prime=0.0))
    batch = _batch()
    actor_before = _copies(algorithm.actor)
    critic_one_before = _copies(algorithm.critic_one)

    next_actions = algorithm.actor(batch.next_observations).detach()
    z_next = algorithm.critic_two(batch.next_observations, next_actions)
    target = (batch.rewards + algorithm.settings.gamma * batch.bootstrap_mask * z_next).detach()
    z_current = algorithm.critic_two(batch.observations, batch.actions)
    reference_loss = 0.5 * (z_current - target).square().mean()
    expected_gradients = torch.autograd.grad(reference_loss, tuple(algorithm.critic_two.parameters()))

    wrong_z_next = algorithm.critic_two(batch.next_observations, next_actions)
    wrong_target = batch.rewards + algorithm.settings.gamma * batch.bootstrap_mask * wrong_z_next
    wrong_z_current = algorithm.critic_two(batch.observations, batch.actions)
    wrong_loss = 0.5 * (wrong_z_current - wrong_target).square().mean()
    wrong_gradients = torch.autograd.grad(wrong_loss, tuple(algorithm.critic_two.parameters()))
    assert any(
        not torch.allclose(expected, wrong)
        for expected, wrong in zip(expected_gradients, wrong_gradients, strict=True)
    )

    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", lambda: None)
    result = algorithm._sequential_critic_two(batch)

    assert result.loss == pytest.approx(float(reference_loss.detach()))
    assert result.td_loss == pytest.approx(float(reference_loss.detach()))
    _assert_gradients_match(expected_gradients, algorithm.critic_two)
    _assert_unchanged(actor_before, algorithm.actor)
    _assert_unchanged(critic_one_before, algorithm.critic_one)
    assert all(parameter.grad is None for parameter in algorithm.actor.parameters())
    assert all(parameter.grad is None for parameter in algorithm.critic_one.parameters())


def test_q1_recomputes_updated_q2_and_keeps_next_agreement_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(_settings(critic_two_learning_rate=5e-3))
    batch = _batch()
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        z_before = algorithm.critic_two(batch.next_observations, next_actions).clone()
    algorithm._sequential_critic_two(batch)
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        z_next = algorithm.critic_two(batch.next_observations, next_actions)
    assert not torch.equal(z_before, z_next)

    actor_before = _copies(algorithm.actor)
    critic_two_before = _copies(algorithm.critic_two)
    next_actions = algorithm.actor(batch.next_observations).detach()
    with torch.no_grad():
        z_next = algorithm.critic_two(batch.next_observations, next_actions)
        target = batch.rewards + algorithm.settings.gamma * batch.bootstrap_mask * z_next
    y = algorithm.critic_one(batch.observations, batch.actions)
    y_next = algorithm.critic_one(batch.next_observations, next_actions)
    td_loss = 0.5 * (y - target).square().mean()
    next_agreement = 0.5 * (y_next - z_next).square().mean()
    reference_loss = td_loss + algorithm.settings.alpha * next_agreement
    expected_gradients = torch.autograd.grad(
        reference_loss, tuple(algorithm.critic_one.parameters()), retain_graph=True
    )
    td_only_gradients = torch.autograd.grad(td_loss, tuple(algorithm.critic_one.parameters()))
    assert any(
        not torch.allclose(expected, td_only)
        for expected, td_only in zip(expected_gradients, td_only_gradients, strict=True)
    )

    monkeypatch.setattr(algorithm.critic_one_optimizer, "step", lambda: None)
    result = algorithm._sequential_critic_one(batch)

    assert result.loss == pytest.approx(float(reference_loss.detach()))
    assert result.td_loss == pytest.approx(float(td_loss.detach()))
    assert result.agreement_loss == pytest.approx(float(next_agreement.detach()))
    _assert_gradients_match(expected_gradients, algorithm.critic_one)
    _assert_unchanged(actor_before, algorithm.actor)
    _assert_unchanged(critic_two_before, algorithm.critic_two)
    assert all(parameter.grad is None for parameter in algorithm.actor.parameters())
    assert all(parameter.grad is None for parameter in algorithm.critic_two.parameters())


def test_actor_gradient_passes_through_frozen_q2_action_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    batch = _batch()
    critic_one_before = _copies(algorithm.critic_one)
    critic_two_before = _copies(algorithm.critic_two)
    for parameter in (*tuple(algorithm.critic_one.parameters()), *tuple(algorithm.critic_two.parameters())):
        parameter.requires_grad_(False)
    reference_loss = -algorithm.critic_two(
        batch.next_observations, algorithm.actor(batch.next_observations)
    ).mean()
    expected_gradients = torch.autograd.grad(reference_loss, tuple(algorithm.actor.parameters()))
    for parameter in (*tuple(algorithm.critic_one.parameters()), *tuple(algorithm.critic_two.parameters())):
        parameter.requires_grad_(True)
    assert any(torch.count_nonzero(gradient).item() for gradient in expected_gradients)

    monkeypatch.setattr(algorithm.actor_optimizer, "step", lambda: None)
    result = algorithm._sequential_actor(batch)

    assert result.loss == pytest.approx(float(reference_loss.detach()))
    _assert_gradients_match(expected_gradients, algorithm.actor)
    _assert_unchanged(critic_one_before, algorithm.critic_one)
    _assert_unchanged(critic_two_before, algorithm.critic_two)
    assert all(parameter.grad is None for parameter in algorithm.critic_one.parameters())
    assert all(parameter.grad is None for parameter in algorithm.critic_two.parameters())


def test_separate_td_metrics_include_both_residuals_and_phase_losses() -> None:
    algorithm = ThreeNetworkDDPG(_settings())
    batch = _batch()
    snapshot = algorithm._snapshot(batch)
    expected_prime = q2_td_residual(
        batch.rewards,
        batch.bootstrap_mask,
        snapshot.z,
        snapshot.z1,
        algorithm.settings.gamma,
    )
    expected_shared = shared_residual(
        batch.rewards,
        batch.bootstrap_mask,
        snapshot.z,
        snapshot.y,
        algorithm.settings.gamma,
    )
    torch.testing.assert_close(snapshot.delta_prime, expected_prime)
    torch.testing.assert_close(snapshot.delta, expected_shared)

    metrics = {record.name: record.value for record in algorithm.update_from_batch(batch).metrics}
    assert metrics["shared_squared_error"] == pytest.approx(float(shared_squared_error(expected_shared)))
    assert metrics["q2_td_squared_error"] == pytest.approx(float(shared_squared_error(expected_prime)))
    assert "q2_signed_residual_mean" in metrics
    assert "q2_absolute_residual_mean" in metrics
    assert "critic_one_td_loss" in metrics
    assert "critic_two_td_loss" in metrics


def test_separate_td_configuration_is_explicit_and_restricted() -> None:
    config = load_config(
        algorithm="three_network_ddpg",
        overrides=["algorithm.critic_objective=separate_td"],
    )
    assert config.algorithm.critic_objective == "separate_td"
    with pytest.raises(ValueError, match="sequential update schedule"):
        _settings(update_schedule="simultaneous")
    with pytest.raises(ValueError, match="beta regularization to be disabled"):
        _settings(beta_mode="l2_y", beta_coefficient=0.1)


def test_manual_screening_manifest_is_bounded_and_paired() -> None:
    manifest = load_manifest(MANIFEST)
    planned = expand_manifest(manifest)

    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert manifest.output_root == Path("runs/pendulum-separate-td-screening-01")
    assert manifest.workers == 1
    assert len(planned) == 3
    assert {run.variant_id for run in planned} == {
        "ddpg",
        "three-network-shared-error",
        "three-network-separate-td",
    }
    assert all(run.training_seed == 11 for run in planned)
    assert all(run.evaluation_seeds == manifest.evaluation_seeds for run in planned)
    assert all(int(run.config.experiment.total_steps) == 5000 for run in planned)
    assert all(int(run.config.experiment.evaluation_frequency) == 500 for run in planned)

    proposed = {run.variant_id: run for run in planned if run.config.algorithm.name == "three_network_ddpg"}
    assert proposed["three-network-shared-error"].config.algorithm.critic_objective == "shared_error"
    assert proposed["three-network-separate-td"].config.algorithm.critic_objective == "separate_td"
    for run in proposed.values():
        algorithm = run.config.algorithm
        assert algorithm.alpha == pytest.approx(0.1)
        assert algorithm.alpha_prime == pytest.approx(0.1)
        assert algorithm.agreement_mode == "both"
        assert algorithm.beta_regularizer.mode == "none"
        assert algorithm.beta_regularizer.coefficient == 0.0
        assert algorithm.update_schedule == "sequential"
        assert algorithm.hidden_layers == [128, 128]
        assert algorithm.diagnostic_frequency == 100
