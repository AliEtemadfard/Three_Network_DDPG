"""Controlled-ablation tests for the three-network research method."""

from copy import deepcopy
from typing import Any

import pytest
import torch

from three_network_ddpg.algorithms.three_network_ddpg import (
    ThreeNetworkDDPG,
    ThreeNetworkSettings,
    agreement_error,
    beta_regularizer,
    shared_residual,
    shared_squared_error,
)
from three_network_ddpg.config import load_config
from three_network_ddpg.core.batches import TransitionBatch
from three_network_ddpg.core.replay import ReplayBuffer
from three_network_ddpg.environments import TinyContinuousEnv
from three_network_ddpg.training import OffPolicyTrainer, TrainerSettings


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
        "agreement_mode": "none",
        "alpha": 0.0,
        "alpha_prime": 0.0,
        "beta_mode": "none",
        "beta_coefficient": 0.0,
        "update_schedule": "sequential",
        "guard_scope": "joint",
        "guard_maximum_retries": 1,
        "guard_minimum_learning_rate": 1e-7,
        "diagnostic_frequency": None,
        "seed": 41,
    }
    values.update(changes)
    return ThreeNetworkSettings(**values)


def _batch(seed: int = 5, batch_size: int = 6) -> TransitionBatch:
    generator = torch.Generator().manual_seed(seed)
    flags = torch.zeros((batch_size, 1), dtype=torch.bool)
    return TransitionBatch(
        observations=torch.randn((batch_size, 3), generator=generator),
        actions=torch.empty((batch_size, 1)).uniform_(-2.0, 2.0, generator=generator),
        rewards=torch.randn((batch_size, 1), generator=generator),
        next_observations=torch.randn((batch_size, 3), generator=generator),
        terminated=flags,
        truncated=flags.clone(),
        bootstrap_mask=torch.ones((batch_size, 1)),
    )


def _parameter_copies(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_equal(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return all(
        torch.equal(previous, current) for previous, current in zip(before, module.parameters(), strict=True)
    )


def _all_gradients_none(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_beta_regularizer_formulas() -> None:
    y = torch.tensor([[2.0], [-1.0]])
    z = torch.tensor([[3.0], [4.0]])
    torch.testing.assert_close(beta_regularizer("none", y, z), torch.tensor(0.0))
    torch.testing.assert_close(beta_regularizer("linear_y", y, z), torch.tensor(0.5))
    torch.testing.assert_close(beta_regularizer("l2_y", y, z), torch.tensor(1.25))


def test_l2_y_z_beta_mode_is_rejected_as_undefined() -> None:
    with pytest.raises(ValueError, match="symmetric Y/Z regularization is not defined"):
        _settings(beta_mode="l2_y_z", beta_coefficient=0.1)


@pytest.mark.parametrize("mode", ["current", "both"])
def test_current_agreement_uses_detached_y_only_in_critic_two(mode: str) -> None:
    settings = _settings(
        agreement_mode=mode,
        alpha=0.2 if mode == "both" else 0.0,
        alpha_prime=0.7,
    )
    algorithm = ThreeNetworkDDPG(settings)
    batch = _batch()
    actor_before = _parameter_copies(algorithm.actor)
    critic_one_before = _parameter_copies(algorithm.critic_one)
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        y = algorithm.critic_one(batch.observations, batch.actions)
        z = algorithm.critic_two(batch.next_observations, next_actions)
        z1 = algorithm.critic_two(batch.observations, batch.actions)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z, y, settings.gamma)
        )
        current = agreement_error(z1, y)
        expected = shared + settings.alpha_prime * current

    result = algorithm._sequential_critic_two(batch)
    assert result.loss == pytest.approx(float(expected))
    assert result.agreement_loss == pytest.approx(float(current))
    assert _parameters_equal(actor_before, algorithm.actor)
    assert _parameters_equal(critic_one_before, algorithm.critic_one)
    assert _all_gradients_none(algorithm.actor)
    assert _all_gradients_none(algorithm.critic_one)


@pytest.mark.parametrize("mode", ["next", "both"])
def test_next_agreement_uses_detached_z_only_in_critic_one(mode: str) -> None:
    settings = _settings(
        agreement_mode=mode,
        alpha=0.6,
        alpha_prime=0.2 if mode == "both" else 0.0,
    )
    algorithm = ThreeNetworkDDPG(settings)
    batch = _batch()
    actor_before = _parameter_copies(algorithm.actor)
    critic_two_before = _parameter_copies(algorithm.critic_two)
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        z = algorithm.critic_two(batch.next_observations, next_actions)
        y = algorithm.critic_one(batch.observations, batch.actions)
        y1 = algorithm.critic_one(batch.next_observations, next_actions)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z, y, settings.gamma)
        )
        next_error = agreement_error(y1, z)
        expected = shared + settings.alpha * next_error

    result = algorithm._sequential_critic_one(batch)
    assert result.loss == pytest.approx(float(expected))
    assert result.agreement_loss == pytest.approx(float(next_error))
    assert _parameters_equal(actor_before, algorithm.actor)
    assert _parameters_equal(critic_two_before, algorithm.critic_two)
    assert _all_gradients_none(algorithm.actor)
    assert _all_gradients_none(algorithm.critic_two)


@pytest.mark.parametrize("mode", ["linear_y", "l2_y"])
def test_beta_term_is_applied_to_critic_one_formula(mode: str) -> None:
    settings = _settings(beta_mode=mode, beta_coefficient=0.3)
    algorithm = ThreeNetworkDDPG(settings)
    batch = _batch()
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        z = algorithm.critic_two(batch.next_observations, next_actions)
        y = algorithm.critic_one(batch.observations, batch.actions)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z, y, settings.gamma)
        )
        beta = beta_regularizer(mode, y, z)
        expected = shared + settings.beta_coefficient * beta
    result = algorithm._sequential_critic_one(batch)
    assert result.beta_loss == pytest.approx(float(beta))
    assert result.loss == pytest.approx(float(expected))


def test_simultaneous_gradients_match_one_preupdate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        update_schedule="simultaneous",
        agreement_mode="both",
        alpha=0.3,
        alpha_prime=0.4,
        beta_mode="l2_y",
        beta_coefficient=0.2,
    )
    algorithm = ThreeNetworkDDPG(settings)
    batch = _batch()
    next_actions = algorithm.actor(batch.next_observations).detach()
    y = algorithm.critic_one(batch.observations, batch.actions)
    y1 = algorithm.critic_one(batch.next_observations, next_actions)
    z1 = algorithm.critic_two(batch.observations, batch.actions)
    z = algorithm.critic_two(batch.next_observations, next_actions)

    one_loss = shared_squared_error(
        shared_residual(batch.rewards, batch.bootstrap_mask, z.detach(), y, settings.gamma)
    )
    one_loss = (
        one_loss
        + settings.alpha * agreement_error(y1, z.detach())
        + settings.beta_coefficient * beta_regularizer(settings.beta_mode, y, z.detach())
    )
    expected_one = torch.autograd.grad(one_loss, tuple(algorithm.critic_one.parameters()))

    two_loss = shared_squared_error(
        shared_residual(batch.rewards, batch.bootstrap_mask, z, y.detach(), settings.gamma)
    )
    two_loss = two_loss + settings.alpha_prime * agreement_error(z1, y.detach())
    expected_two = torch.autograd.grad(two_loss, tuple(algorithm.critic_two.parameters()))

    actor_loss = algorithm._actor_loss(batch)
    expected_actor = torch.autograd.grad(actor_loss, tuple(algorithm.actor.parameters()))

    monkeypatch.setattr(algorithm.actor_optimizer, "step", lambda *args, **kwargs: None)
    monkeypatch.setattr(algorithm.critic_one_optimizer, "step", lambda *args, **kwargs: None)
    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", lambda *args, **kwargs: None)
    algorithm._simultaneous_update(batch)

    for expected, parameter in zip(expected_one, algorithm.critic_one.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected)
    for expected, parameter in zip(expected_two, algorithm.critic_two.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected)
    for expected, parameter in zip(expected_actor, algorithm.actor.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected)


def test_sequential_critic_one_recomputes_after_critic_two_step() -> None:
    algorithm = ThreeNetworkDDPG(_settings(critic_two_learning_rate=5e-3))
    batch = _batch()
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        z_before = algorithm.critic_two(batch.next_observations, next_actions).clone()
    algorithm._sequential_critic_two(batch)
    with torch.no_grad():
        next_actions = algorithm.actor(batch.next_observations)
        z_after = algorithm.critic_two(batch.next_observations, next_actions)
        y = algorithm.critic_one(batch.observations, batch.actions)
        expected = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z_after,
                y,
                algorithm.settings.gamma,
            )
        )
    assert not torch.equal(z_before, z_after)
    result = algorithm._sequential_critic_one(batch)
    assert result.shared_loss == pytest.approx(float(expected))


def test_joint_guard_rejection_restores_parameters_and_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope="joint",
            guard_maximum_retries=0,
        )
    )
    batch = _batch()
    algorithm._sequential_update(batch)
    before = deepcopy(algorithm._capture_training_state())
    values = iter((1.0, 2.0, 1.0))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))

    outcome = algorithm._run_guard("joint", batch, None)

    assert not outcome.accepted
    assert not outcome.update_applied
    after = algorithm._capture_training_state()
    _assert_nested_equal(before, after)


def test_guard_backtracks_from_same_snapshot_and_accepts_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope="joint",
            guard_backtracking_factor=0.25,
            guard_maximum_retries=1,
        )
    )
    values = iter((1.0, 2.0, 0.5))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))
    outcome = algorithm._run_guard("joint", _batch(), None)
    assert outcome.accepted
    assert outcome.termination_reason == "accepted"
    assert outcome.retries == 1
    assert outcome.learning_rate_scale == pytest.approx(0.25)
    assert outcome.effective_learning_rate == pytest.approx(2.5e-4)
    assert outcome.error_before == pytest.approx(1.0)
    assert outcome.error_after == pytest.approx(0.5)
    assert algorithm.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert algorithm.critic_one_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert algorithm.critic_two_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


@pytest.mark.parametrize("scope", ["critics", "actor"])
def test_guard_scope_restores_only_guarded_phase_on_rejection(
    scope: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope=scope,
            guard_maximum_retries=0,
        )
    )
    batch = _batch()
    actor_before = _parameter_copies(algorithm.actor)
    one_before = _parameter_copies(algorithm.critic_one)
    two_before = _parameter_copies(algorithm.critic_two)
    values = iter((1.0, 2.0, 1.0))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))
    outcome = algorithm._guarded_update(batch, None)
    assert not outcome.accepted
    assert outcome.update_applied
    if scope == "critics":
        assert not _parameters_equal(actor_before, algorithm.actor)
        assert _parameters_equal(one_before, algorithm.critic_one)
        assert _parameters_equal(two_before, algorithm.critic_two)
    else:
        assert _parameters_equal(actor_before, algorithm.actor)
        assert not _parameters_equal(one_before, algorithm.critic_one)
        assert not _parameters_equal(two_before, algorithm.critic_two)


def test_guard_validation_batch_is_required_and_logged() -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope="joint",
            guard_tolerance=1e9,
            guard_validation_batch=True,
        )
    )
    assert algorithm.requires_validation_batch
    with pytest.raises(ValueError, match="requires a validation batch"):
        algorithm.update_from_batch(_batch())
    result = algorithm.update_from_batch(_batch(), validation_batch=_batch(seed=19))
    names = {metric.name for metric in result.metrics}
    assert "guard_validation_error_change" in names
    assert "guard_accepted_step" in names
    assert "guard_retry_count" in names
    assert "guard_effective_learning_rate" in names


def test_alignment_diagnostics_run_only_at_configured_interval() -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            agreement_mode="both",
            alpha=0.2,
            alpha_prime=0.2,
            beta_mode="l2_y",
            beta_coefficient=0.1,
            diagnostic_frequency=2,
        )
    )
    first = {metric.name for metric in algorithm.update_from_batch(_batch()).metrics}
    second = {metric.name for metric in algorithm.update_from_batch(_batch()).metrics}
    for prefix in ("actor", "critic_one", "critic_two"):
        assert f"{prefix}_shared_gradient_inner_product" not in first
        assert f"{prefix}_shared_gradient_inner_product" in second
        assert f"{prefix}_shared_gradient_cosine" in second


def test_trainer_draws_validation_batch_for_guarded_monitoring() -> None:
    environment = TinyContinuousEnv(horizon=4)
    try:
        algorithm = ThreeNetworkDDPG(
            ThreeNetworkSettings(
                observation_dim=1,
                action_dim=1,
                action_low=(-2.0,),
                action_high=(2.0,),
                hidden_layers=(8, 8),
                update_schedule="guarded",
                guard_scope="joint",
                guard_tolerance=1e9,
                guard_validation_batch=True,
                diagnostic_frequency=None,
                seed=7,
            )
        )
        replay = ReplayBuffer(
            capacity=16,
            observation_shape=(1,),
            action_shape=(1,),
            seed=7,
            bootstrap_on_truncation=True,
            device="cpu",
        )
        summary = OffPolicyTrainer(
            algorithm=algorithm,
            environment=environment,
            replay=replay,
            settings=TrainerSettings(8, 2, 2, True, True),
        ).run(seed=7, deterministic_algorithms=True)
        assert "guard_validation_error_change" in {metric.name for metric in summary.metrics}
    finally:
        environment.close()


def test_guard_first_proposal_metrics_are_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(_settings(update_schedule="guarded", guard_scope="joint"))
    values = iter((1.0, 0.5))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))

    outcome = algorithm._run_guard("joint", _batch(), None)

    assert outcome.accepted
    assert outcome.termination_reason == "accepted"
    assert outcome.retries == 0
    assert outcome.learning_rate_scale == pytest.approx(1.0)
    assert outcome.effective_learning_rate == pytest.approx(1e-3)
    assert outcome.error_before == pytest.approx(1.0)
    assert outcome.error_after == pytest.approx(0.5)


def test_guard_all_attempted_proposals_rejected_reports_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope="joint",
            guard_backtracking_factor=0.5,
            guard_maximum_retries=2,
        )
    )
    values = iter((1.0, 2.0, 1.8, 1.4, 1.0))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))

    outcome = algorithm._run_guard("joint", _batch(), None)

    assert not outcome.accepted
    assert outcome.termination_reason == "retries_exhausted"
    assert outcome.retries == 2
    assert outcome.learning_rate_scale == pytest.approx(0.25)
    assert outcome.effective_learning_rate == pytest.approx(2.5e-4)
    assert outcome.error_before == pytest.approx(1.0)
    assert outcome.error_after == pytest.approx(1.4)
    assert outcome.committed_error_after == pytest.approx(1.0)


def test_guard_minimum_rate_reports_previous_executed_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope="joint",
            guard_backtracking_factor=0.1,
            guard_maximum_retries=2,
            guard_minimum_learning_rate=5e-4,
        )
    )
    values = iter((1.0, 2.0, 1.0))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))

    outcome = algorithm._run_guard("joint", _batch(), None)

    assert not outcome.accepted
    assert outcome.termination_reason == "minimum_learning_rate"
    assert outcome.retries == 0
    assert outcome.learning_rate_scale == pytest.approx(1.0)
    assert outcome.effective_learning_rate == pytest.approx(1e-3)
    assert outcome.error_before == pytest.approx(1.0)
    assert outcome.error_after == pytest.approx(2.0)
    assert outcome.committed_error_after == pytest.approx(1.0)


def test_three_network_config_rejects_undefined_l2_y_z_mode() -> None:
    with pytest.raises(ValueError, match="separately approved objective"):
        load_config(
            algorithm="three_network_ddpg",
            overrides=[
                "algorithm.beta_regularizer.mode=l2_y_z",
                "algorithm.beta_regularizer.coefficient=0.1",
            ],
        )


def test_three_network_from_config_resolves_auto_before_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        algorithm="three_network_ddpg",
        overrides=["algorithm.device=auto"],
    )
    algorithm = ThreeNetworkDDPG.from_config(
        config.algorithm,
        observation_dim=3,
        action_low=torch.tensor([-2.0]).numpy(),
        action_high=torch.tensor([2.0]).numpy(),
    )
    assert algorithm.settings.device == "cpu"


def test_guard_metric_record_fields_describe_one_executed_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = ThreeNetworkDDPG(
        _settings(
            update_schedule="guarded",
            guard_scope="joint",
            guard_backtracking_factor=0.25,
            guard_maximum_retries=1,
        )
    )
    values = iter((1.0, 2.0, 0.5))
    monkeypatch.setattr(algorithm, "_shared_error_value", lambda _: next(values))

    result = algorithm.update_from_batch(_batch())
    metrics = {metric.name: metric.value for metric in result.metrics}

    assert result.updates_applied == 1
    assert metrics["guard_accepted_step"] == 1.0
    assert metrics["guard_rejected_step"] == 0.0
    assert metrics["guard_retry_count"] == 1.0
    assert metrics["guard_learning_rate_scale"] == pytest.approx(0.25)
    assert metrics["guard_effective_learning_rate"] == pytest.approx(2.5e-4)
    assert metrics["guard_error_before"] == pytest.approx(1.0)
    assert metrics["guard_error_after"] == pytest.approx(0.5)
    assert metrics["guard_stopped_at_minimum_learning_rate"] == 0.0
    assert metrics["guard_retries_exhausted"] == 0.0
