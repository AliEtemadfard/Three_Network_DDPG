"""Independent equation/gradient and schedule checks; no environment interaction."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from three_network_ddpg.algorithms.registry import create_algorithm
from three_network_ddpg.algorithms.shared_d import (
    BoardSharedDDDPG,
    BoardSharedDSettings,
    ConditionedSharedDDDPG,
    ConditionedSharedDSettings,
    SharedDDDPG,
    SharedDSettings,
    SynchronousSharedDDDPG,
    SynchronousSharedDSettings,
    TwoTimescaleSharedDDDPG,
    TwoTimescaleSharedDSettings,
)
from three_network_ddpg.algorithms.three_network_ddpg import ThreeNetworkDDPG, ThreeNetworkSettings
from three_network_ddpg.config import load_config
from three_network_ddpg.core.batches import TransitionBatch
from three_network_ddpg.protocol.manifest import expand_manifest, load_manifest


def settings(**changes) -> SharedDSettings:
    return SharedDSettings(
        observation_dim=3,
        action_dim=1,
        action_low=(-2.0,),
        action_high=(2.0,),
        hidden_layers=(8, 8),
        seed=47,
        diagnostic_frequency=None,
        **changes,
    )


def board_settings(**changes) -> BoardSharedDSettings:
    values = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
        "seed": 47,
        "diagnostic_frequency": None,
    }
    values.update(changes)
    return BoardSharedDSettings(**values)


def synchronous_settings(**changes) -> SynchronousSharedDSettings:
    values = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
        "seed": 47,
        "diagnostic_frequency": None,
        "critic_objective": "shared_error",
        "agreement_mode": "both",
        "alpha": 0.1,
        "alpha_prime": 0.1,
        "update_schedule": "simultaneous",
        "actor_learning_starts": 5_000,
        "actor_update_delay": 2,
    }
    values.update(changes)
    return SynchronousSharedDSettings(**values)


def batch() -> TransitionBatch:
    generator = torch.Generator().manual_seed(9)
    flags = torch.zeros((7, 1), dtype=torch.bool)
    mask = torch.ones((7, 1))
    mask[-1] = 0.0
    return TransitionBatch(
        observations=torch.randn((7, 3), generator=generator),
        actions=torch.randn((7, 1), generator=generator),
        rewards=torch.randn((7, 1), generator=generator),
        next_observations=torch.randn((7, 3), generator=generator),
        terminated=flags,
        truncated=flags.clone(),
        bootstrap_mask=mask,
    )


def copies(module):
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def unchanged(before, module):
    for original, parameter in zip(before, module.parameters(), strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["one", "two"])
def test_critic_losses_and_gradients_match_independent_shared_d_reference(monkeypatch, phase):
    algorithm = SharedDDDPG(settings(alpha=0.3, alpha_prime=0.4))
    data = batch()
    a_next = algorithm.actor(data.next_observations).detach()
    if phase == "two":
        y = algorithm.critic_one(data.observations, data.actions).detach()
        z = algorithm.critic_two(data.observations, data.actions)
        z_next = algorithm.critic_two(data.next_observations, a_next)
        d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
        reference = 0.5 * d.square().mean() + 0.5 * algorithm.settings.alpha_prime * (z - y).square().mean()
        module = algorithm.critic_two
    else:
        z_next = algorithm.critic_two(data.next_observations, a_next).detach()
        y = algorithm.critic_one(data.observations, data.actions)
        y_next = algorithm.critic_one(data.next_observations, a_next)
        d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
        reference = (
            0.5 * d.square().mean() + 0.5 * algorithm.settings.alpha * (y_next - z_next).square().mean()
        )
        module = algorithm.critic_one
    expected = torch.autograd.grad(reference, tuple(module.parameters()))
    monkeypatch.setattr(getattr(algorithm, f"critic_{phase}_optimizer"), "step", lambda: None)
    result = getattr(algorithm, f"_sequential_critic_{phase}")(data)
    assert result.loss == pytest.approx(float(reference.detach()))
    for gradient, parameter in zip(expected, module.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_zero_agreement_q2_trains_next_bootstrap_not_current_td(monkeypatch):
    algorithm = SharedDDDPG(settings(alpha=0.0, alpha_prime=0.0))
    data = batch()
    a_next = algorithm.actor(data.next_observations).detach()
    y = algorithm.critic_one(data.observations, data.actions).detach()
    z_next = algorithm.critic_two(data.next_observations, a_next)
    z_current = algorithm.critic_two(data.observations, data.actions)
    d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
    reference = 0.5 * d.square().mean()
    next_gradient, current_gradient = torch.autograd.grad(
        reference, (z_next, z_current), allow_unused=True, retain_graph=True
    )
    torch.testing.assert_close(next_gradient, algorithm.settings.gamma * data.bootstrap_mask * d / 7)
    assert torch.count_nonzero(next_gradient).item() > 0
    assert next_gradient[-1] == 0  # Terminal samples cannot train Q2 through bootstrap.
    assert current_gradient is None
    expected = torch.autograd.grad(reference, tuple(algorithm.critic_two.parameters()))
    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", lambda: None)
    algorithm._sequential_critic_two(data)
    for gradient, parameter in zip(expected, algorithm.critic_two.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.parametrize("phase", ["one", "two"])
def test_each_critic_phase_updates_only_its_own_parameters(phase):
    algorithm = SharedDDDPG(settings())
    actor_before = copies(algorithm.actor)
    other = algorithm.critic_two if phase == "one" else algorithm.critic_one
    own = getattr(algorithm, f"critic_{phase}")
    other_before, own_before = copies(other), copies(own)
    getattr(algorithm, f"_sequential_critic_{phase}")(batch())
    unchanged(actor_before, algorithm.actor)
    unchanged(other_before, other)
    assert any(
        not torch.equal(before, after) for before, after in zip(own_before, own.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in (*algorithm.actor.parameters(), *other.parameters()))


def test_q1_recomputes_q2_and_retains_next_agreement_gradient(monkeypatch):
    algorithm = SharedDDDPG(settings(alpha=0.5))
    data = batch()
    with torch.no_grad():
        a_next = algorithm.actor(data.next_observations)
        stale = algorithm.critic_two(data.next_observations, a_next).clone()
    algorithm._sequential_critic_two(data)
    with torch.no_grad():
        fresh = algorithm.critic_two(data.next_observations, a_next)
    assert not torch.equal(stale, fresh)
    y = algorithm.critic_one(data.observations, data.actions)
    y_next = algorithm.critic_one(data.next_observations, a_next)
    td = 0.5 * (data.rewards + algorithm.settings.gamma * data.bootstrap_mask * fresh - y).square().mean()
    agreement = 0.5 * (y_next - fresh).square().mean()
    parameters = tuple(algorithm.critic_one.parameters())
    expected = torch.autograd.grad(td + algorithm.settings.alpha * agreement, parameters, retain_graph=True)
    td_only = torch.autograd.grad(td, parameters)
    assert any(not torch.allclose(a, b) for a, b in zip(expected, td_only, strict=True))
    monkeypatch.setattr(algorithm.critic_one_optimizer, "step", lambda: None)
    algorithm._sequential_critic_one(data)
    for gradient, parameter in zip(expected, parameters, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_actor_receives_action_gradient_with_both_critics_fixed(monkeypatch):
    algorithm = SharedDDDPG(settings())
    data = batch()
    critic_one_before, critic_two_before = copies(algorithm.critic_one), copies(algorithm.critic_two)
    critic_parameters = (*algorithm.critic_one.parameters(), *algorithm.critic_two.parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    reference = -algorithm.critic_two(data.next_observations, algorithm.actor(data.next_observations)).mean()
    expected = torch.autograd.grad(reference, tuple(algorithm.actor.parameters()))
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    assert any(torch.count_nonzero(gradient).item() for gradient in expected)
    monkeypatch.setattr(algorithm.actor_optimizer, "step", lambda: None)
    algorithm._sequential_actor(data)
    for gradient, parameter in zip(expected, algorithm.actor.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    unchanged(critic_one_before, algorithm.critic_one)
    unchanged(critic_two_before, algorithm.critic_two)
    assert all(parameter.grad is None for parameter in critic_parameters)


def test_delayed_update_order_and_checkpoint_clock(monkeypatch):
    algorithm = SharedDDDPG(settings())
    calls = []
    for name in ("critic_two", "critic_one", "actor"):
        original = getattr(algorithm, f"_sequential_{name}")

        def recording(data, name=name, original=original):
            calls.append(name)
            return original(data)

        monkeypatch.setattr(algorithm, f"_sequential_{name}", recording)
    actor_before = copies(algorithm.actor)
    first = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
    assert calls == ["critic_two", "critic_one"]
    unchanged(actor_before, algorithm.actor)
    assert first["actor_update_applied"] == 0
    assert first["actor_update_steps"] == 0
    assert "actor_gradient_norm" not in first
    assert not algorithm.actor_optimizer.state

    resumed = SharedDDDPG(settings())
    resumed.load_state_dict(deepcopy(algorithm.state_dict()))
    second = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
    resumed_second = {record.name: record.value for record in resumed.update_from_batch(batch()).metrics}
    assert second == resumed_second
    assert calls == ["critic_two", "critic_one", "critic_two", "critic_one", "actor"]
    assert second["actor_update_applied"] == 1
    assert second["actor_update_steps"] == 1
    for name in ("actor", "critic_one", "critic_two"):
        unchanged(copies(getattr(algorithm, name)), getattr(resumed, name))
    with pytest.raises(ValueError, match="settings do not match"):
        SharedDDDPG(settings(actor_update_delay=3)).load_state_dict(algorithm.state_dict())


def test_delay_one_exactly_preserves_historical_shared_error_updates():
    configured = settings(actor_update_delay=1)
    previous = configured.to_dict()
    del previous["actor_update_delay"]
    old = ThreeNetworkDDPG(ThreeNetworkSettings(**previous))
    new = SharedDDDPG(configured)
    for _ in range(2):
        old_metrics = {record.name: record.value for record in old.update_from_batch(batch()).metrics}
        new_metrics = {record.name: record.value for record in new.update_from_batch(batch()).metrics}
        assert all(new_metrics[key] == value for key, value in old_metrics.items())
        for name in ("actor", "critic_one", "critic_two"):
            unchanged(copies(getattr(old, name)), getattr(new, name))
    assert "actor_update_delay" not in old.settings.to_dict()


def test_registry_configuration_and_architecture():
    config = load_config(algorithm="three_network_shared_d", experiment="three_network_core")
    algorithm = create_algorithm(
        "three_network_shared_d",
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    assert isinstance(algorithm, SharedDDDPG)
    assert algorithm.capabilities.name == "three_network_shared_d"
    assert algorithm.settings.actor_update_delay == 2
    assert "target_networks" not in config.algorithm
    assert not hasattr(algorithm, "target_actor")
    assert not hasattr(algorithm, "target_critic")
    assert {id(parameter) for parameter in algorithm.critic_one.parameters()}.isdisjoint(
        id(parameter) for parameter in algorithm.critic_two.parameters()
    )
    assert any(
        not torch.equal(one, two)
        for one, two in zip(algorithm.critic_one.parameters(), algorithm.critic_two.parameters(), strict=True)
    )


def test_proof_residual_identities_hold_without_using_d_prime_as_a_loss():
    algorithm = SharedDDDPG(settings())
    data = batch()
    snapshot = algorithm._snapshot(data)
    torch.testing.assert_close(snapshot.delta_prime, snapshot.delta + snapshot.y - snapshot.z1)
    q1_residual = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * snapshot.y1 - snapshot.y
    torch.testing.assert_close(
        q1_residual,
        snapshot.delta + algorithm.settings.gamma * data.bootstrap_mask * (snapshot.y1 - snapshot.z),
    )


@pytest.mark.parametrize(
    "override",
    [
        "algorithm.actor_update_delay=0",
        "algorithm.actor_update_delay=1.5",
        "algorithm.critic_objective=separate_td",
        "algorithm.agreement_mode=none",
        "algorithm.beta_regularizer.mode=l2_y_z",
        "algorithm.update_schedule=guarded",
        "algorithm.actor_state_source=current",
        "algorithm.target_networks.actor=true",
    ],
)
def test_incompatible_configurations_are_rejected(override):
    with pytest.raises(ValueError):
        load_config(algorithm="three_network_shared_d", overrides=[override])


def test_long_manual_manifest_is_paired_isolated_and_fixed():
    repository = Path(__file__).resolve().parents[2]
    path = repository / "experiments/manifests/pendulum_shared_d_long_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert len(planned) == 2
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert manifest.workers == 1
    assert manifest.output_root == Path("runs/pendulum-shared-d-long-01")
    assert len({run.run_id for run in planned}) == 2
    assert [run.run_id for run in planned] == [run.run_id for run in expand_manifest(load_manifest(path))]
    baseline, proposed = planned
    assert baseline.config.algorithm.name == "ddpg"
    assert proposed.config.algorithm.name == "three_network_shared_d"
    assert sum(run.config.experiment.total_steps for run in planned) == 200000
    assert all(run.config.experiment.total_steps == 100000 for run in planned)
    assert all(
        run.training_seed == 11 and run.evaluation_seeds == manifest.evaluation_seeds for run in planned
    )
    assert set(manifest.training_seeds).isdisjoint(manifest.evaluation_seeds)
    confirmatory = load_manifest(repository / "experiments/manifests/confirmatory.yaml")
    assert set(manifest.training_seeds).isdisjoint(confirmatory.training_seeds)
    assert baseline.config.environment == proposed.config.environment
    for key in (
        "total_steps",
        "warmup_steps",
        "batch_size",
        "replay_capacity",
        "evaluation_frequency",
        "evaluation_episodes",
        "bootstrap_on_truncation",
        "clip_actions",
        "tracking",
    ):
        assert baseline.config.experiment[key] == proposed.config.experiment[key]
    for key in (
        "hidden_layers",
        "activation",
        "gamma",
        "actor_learning_rate",
        "critic_one_learning_rate",
        "optimizer",
        "exploration_noise",
        "action_rescaling",
        "gradient_clip_norm",
        "device",
    ):
        assert baseline.config.algorithm[key] == proposed.config.algorithm[key]
    assert proposed.config.algorithm.critic_objective == "shared_error"
    assert proposed.config.algorithm.agreement_mode == "both"
    assert proposed.config.algorithm.alpha == proposed.config.algorithm.alpha_prime == 0.1
    assert proposed.config.algorithm.beta_regularizer.mode == "none"
    assert proposed.config.algorithm.beta_regularizer.coefficient == 0
    assert proposed.config.algorithm.actor_update_delay == 2
    assert "target_networks" not in proposed.config.algorithm
    assert not proposed.config.algorithm.guard.validation_batch


def conditioned_settings(**changes) -> ConditionedSharedDSettings:
    values = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
        "seed": 47,
        "diagnostic_frequency": None,
        "actor_learning_starts": 3,
    }
    values.update(changes)
    return ConditionedSharedDSettings(**values)


def test_conditioned_q2_loss_and_gradients_keep_shared_d_and_strong_current_agreement(monkeypatch):
    algorithm = ConditionedSharedDDDPG(conditioned_settings())
    data = batch()
    a_next = algorithm.actor(data.next_observations).detach()
    y = algorithm.critic_one(data.observations, data.actions).detach()
    z = algorithm.critic_two(data.observations, data.actions)
    z_next = algorithm.critic_two(data.next_observations, a_next)
    d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
    shared = 0.5 * d.square().mean()
    agreement = 0.5 * (z - y).square().mean()
    reference = shared + algorithm.settings.alpha_prime * agreement
    expected = torch.autograd.grad(reference, tuple(algorithm.critic_two.parameters()))
    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", lambda: None)
    result = algorithm._sequential_critic_two(data)
    assert result.loss == pytest.approx(float(reference.detach()))
    assert result.shared_loss == pytest.approx(float(shared.detach()))
    assert result.agreement_loss == pytest.approx(float(agreement.detach()))
    for gradient, parameter in zip(expected, algorithm.critic_two.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_conditioned_coefficients_improve_the_local_shared_d_slow_mode():
    gamma = 0.99

    def eigenvalue_magnitudes(alpha, alpha_prime):
        matrix = torch.tensor(
            [
                [-(1 + alpha), gamma + alpha],
                [gamma + alpha_prime, -(gamma**2 + alpha_prime)],
            ],
            dtype=torch.float64,
        )
        return torch.linalg.eigvals(matrix).abs().sort().values

    old = eigenvalue_magnitudes(0.1, 0.1)
    corrected = eigenvalue_magnitudes(0.01, 10.0)
    assert old[0].item() == pytest.approx(4.5869552e-6)
    assert corrected[0] > old[0] * 1_800
    assert 10.0 > gamma * 0.01


def test_conditioned_actor_waits_for_critic_preparation_and_checkpoint_clock():
    algorithm = ConditionedSharedDDDPG(conditioned_settings())
    actor_before = copies(algorithm.actor)
    for expected_step in range(1, 4):
        metrics = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
        assert metrics["update_steps"] == expected_step
        assert metrics["actor_update_applied"] == 0
        assert metrics["actor_update_steps"] == 0
        assert "actor_gradient_norm" not in metrics
        unchanged(actor_before, algorithm.actor)

    resumed = ConditionedSharedDDDPG(conditioned_settings())
    resumed.load_state_dict(deepcopy(algorithm.state_dict()))
    fourth = {record.name: record.value for record in resumed.update_from_batch(batch()).metrics}
    assert fourth["actor_update_applied"] == 0
    fifth = {record.name: record.value for record in resumed.update_from_batch(batch()).metrics}
    assert fifth["actor_update_applied"] == 1
    assert fifth["actor_update_steps"] == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, resumed.actor.parameters(), strict=True)
    )


def test_conditioned_registry_and_invalid_coupling_are_explicit():
    config = load_config(algorithm="three_network_shared_d_conditioned")
    algorithm = create_algorithm(
        "three_network_shared_d_conditioned",
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    assert isinstance(algorithm, ConditionedSharedDDDPG)
    assert algorithm.settings.alpha == pytest.approx(0.01)
    assert algorithm.settings.alpha_prime == pytest.approx(10.0)
    assert algorithm.settings.actor_learning_starts == 5000
    assert "target_networks" not in config.algorithm
    with pytest.raises(ValueError, match="alpha_prime"):
        load_config(
            algorithm="three_network_shared_d_conditioned",
            overrides=["algorithm.alpha=0.1", "algorithm.alpha_prime=0.09"],
        )


def test_conditioned_manifest_is_paired_long_and_isolated():
    repository = Path(__file__).resolve().parents[2]
    path = repository / "experiments/manifests/pendulum_shared_d_conditioned_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert len(planned) == 2
    assert manifest.output_root == Path("runs/pendulum-shared-d-conditioned-01")
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert set(manifest.training_seeds).isdisjoint(manifest.evaluation_seeds)
    assert all(run.config.experiment.total_steps == 100000 for run in planned)
    assert {run.config.algorithm.name for run in planned} == {
        "ddpg",
        "three_network_shared_d_conditioned",
    }
    proposed = next(run for run in planned if run.config.algorithm.name.endswith("conditioned"))
    assert proposed.config.algorithm.alpha == pytest.approx(0.01)
    assert proposed.config.algorithm.alpha_prime == pytest.approx(10.0)
    assert proposed.config.algorithm.actor_learning_starts == 5000
    assert proposed.config.algorithm.actor_update_delay == 2
    assert "target_networks" not in proposed.config.algorithm
    confirmatory = load_manifest(repository / "experiments/manifests/confirmatory.yaml")
    assert set(manifest.training_seeds).isdisjoint(confirmatory.training_seeds)


def two_timescale_settings(**changes) -> TwoTimescaleSharedDSettings:
    values = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
        "seed": 47,
        "diagnostic_frequency": None,
        "critic_one_learning_rate": 3e-4,
        "critic_two_learning_rate": 3e-5,
        "actor_learning_starts": 2,
        "critic_two_update_delay": 2,
    }
    values.update(changes)
    return TwoTimescaleSharedDSettings(**values)


def test_two_timescale_board_vectors_use_current_and_next_views_of_one_minibatch(monkeypatch):
    algorithm = TwoTimescaleSharedDDDPG(two_timescale_settings())
    data = batch()
    actor_inputs = []
    critic_one_inputs = []
    critic_two_inputs = []

    def actor_hook(_module, inputs, _output):
        actor_inputs.append(inputs[0].detach().clone())

    def critic_one_hook(_module, inputs, _output):
        critic_one_inputs.append(tuple(value.detach().clone() for value in inputs))

    def critic_two_hook(_module, inputs, _output):
        critic_two_inputs.append(tuple(value.detach().clone() for value in inputs))

    handles = (
        algorithm.actor.register_forward_hook(actor_hook),
        algorithm.critic_one.register_forward_hook(critic_one_hook),
        algorithm.critic_two.register_forward_hook(critic_two_hook),
    )
    monkeypatch.setattr(algorithm.critic_one_optimizer, "step", lambda: None)
    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", lambda: None)
    algorithm._sequential_critic_two(data)
    algorithm._sequential_critic_one(data)
    for handle in handles:
        handle.remove()

    assert actor_inputs
    assert all(torch.equal(value, data.next_observations) for value in actor_inputs)

    current_one = [
        inputs
        for inputs in critic_one_inputs
        if torch.equal(inputs[0], data.observations) and torch.equal(inputs[1], data.actions)
    ]
    next_one = [inputs for inputs in critic_one_inputs if torch.equal(inputs[0], data.next_observations)]
    current_two = [
        inputs
        for inputs in critic_two_inputs
        if torch.equal(inputs[0], data.observations) and torch.equal(inputs[1], data.actions)
    ]
    next_two = [inputs for inputs in critic_two_inputs if torch.equal(inputs[0], data.next_observations)]
    assert len(current_one) == 2  # Detached Y for Q2, then trainable Y for Q1.
    assert len(next_one) == 1  # Y1 = Q1(s_next, actor(s_next)).
    assert len(current_two) == 1  # Z1 = Q2(s, a).
    assert len(next_two) == 2  # Trainable Z for Q2, then detached fresh Z for Q1.
    assert all(inputs[0].shape[0] == data.rewards.shape[0] for inputs in critic_one_inputs)
    assert all(inputs[0].shape[0] == data.rewards.shape[0] for inputs in critic_two_inputs)


def test_two_timescale_update_order_clocks_and_resume_are_exact(monkeypatch):
    algorithm = TwoTimescaleSharedDDDPG(two_timescale_settings())
    calls = []
    for name in ("critic_two", "critic_one", "actor"):
        original = getattr(algorithm, f"_sequential_{name}")

        def recording(data, name=name, original=original):
            calls.append(name)
            return original(data)

        monkeypatch.setattr(algorithm, f"_sequential_{name}", recording)

    expected_calls = [
        ["critic_two", "critic_one"],
        ["critic_one"],
        ["critic_two", "critic_one"],
        ["critic_one"],
        ["critic_two", "critic_one", "actor"],
    ]
    previous = 0
    for step, expected in enumerate(expected_calls, start=1):
        metrics = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
        assert calls[previous:] == expected
        previous = len(calls)
        assert metrics["critic_two_update_applied"] == float(step % 2 == 1)
        assert metrics["critic_two_update_steps"] == (step + 1) // 2
        assert metrics["actor_update_applied"] == float(step == 5)
        assert metrics["actor_update_steps"] == int(step >= 5)

    resumed = TwoTimescaleSharedDDDPG(two_timescale_settings())
    resumed.load_state_dict(deepcopy(algorithm.state_dict()))
    original_metrics = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
    resumed_metrics = {record.name: record.value for record in resumed.update_from_batch(batch()).metrics}
    assert original_metrics == resumed_metrics
    for name in ("actor", "critic_one", "critic_two"):
        unchanged(copies(getattr(algorithm, name)), getattr(resumed, name))


def test_two_timescale_registry_rates_and_no_targets_are_explicit():
    config = load_config(algorithm="three_network_shared_d_two_timescale")
    algorithm = create_algorithm(
        "three_network_shared_d_two_timescale",
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    assert isinstance(algorithm, TwoTimescaleSharedDDDPG)
    assert algorithm.settings.alpha == algorithm.settings.alpha_prime == pytest.approx(0.1)
    assert algorithm.settings.critic_one_learning_rate == pytest.approx(3e-4)
    assert algorithm.settings.critic_two_learning_rate == pytest.approx(3e-5)
    assert algorithm.settings.critic_two_update_delay == 2
    assert algorithm.settings.actor_update_delay == 2
    assert algorithm.settings.actor_learning_starts == 5000
    assert "target_networks" not in config.algorithm
    assert not hasattr(algorithm, "target_actor")
    assert not hasattr(algorithm, "target_critic")


@pytest.mark.parametrize(
    "override",
    [
        "algorithm.critic_two_update_delay=1",
        "algorithm.critic_two_update_delay=2.5",
        "algorithm.critic_two_learning_rate=0.0003",
        "algorithm.actor_learning_starts=-1",
        "algorithm.agreement_mode=current",
        "algorithm.actor_state_source=current",
    ],
)
def test_two_timescale_incompatible_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(algorithm="three_network_shared_d_two_timescale", overrides=[override])


def test_two_timescale_manifest_is_paired_long_and_isolated():
    repository = Path(__file__).resolve().parents[2]
    path = repository / "experiments/manifests/pendulum_shared_d_two_timescale_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert len(planned) == 2
    assert manifest.output_root == Path("runs/pendulum-shared-d-two-timescale-01")
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert set(manifest.training_seeds).isdisjoint(manifest.evaluation_seeds)
    assert all(run.config.experiment.total_steps == 150000 for run in planned)
    assert {run.config.algorithm.name for run in planned} == {
        "ddpg",
        "three_network_shared_d_two_timescale",
    }
    proposed = next(run for run in planned if run.config.algorithm.name.endswith("two_timescale"))
    assert proposed.config.algorithm.alpha == proposed.config.algorithm.alpha_prime == pytest.approx(0.1)
    assert proposed.config.algorithm.critic_one_learning_rate == pytest.approx(3e-4)
    assert proposed.config.algorithm.critic_two_learning_rate == pytest.approx(3e-5)
    assert proposed.config.algorithm.critic_two_update_delay == 2
    assert proposed.config.algorithm.actor_update_delay == 2
    assert proposed.config.algorithm.actor_learning_starts == 5000
    assert proposed.config.algorithm.critic_objective == "shared_error"
    assert proposed.config.algorithm.agreement_mode == "both"
    assert proposed.config.algorithm.beta_regularizer.mode == "none"
    assert not proposed.config.algorithm.guard.validation_batch
    assert "target_networks" not in proposed.config.algorithm
    confirmatory = load_manifest(repository / "experiments/manifests/confirmatory.yaml")
    assert set(manifest.training_seeds).isdisjoint(confirmatory.training_seeds)


def test_board_schedule_updates_both_critics_every_batch_and_delays_only_actor(monkeypatch):
    algorithm = BoardSharedDDDPG(board_settings(actor_learning_starts=2, actor_update_delay=2))
    calls = []
    for name in ("critic_two", "critic_one", "actor"):
        original = getattr(algorithm, f"_sequential_{name}")

        def recording(data, name=name, original=original):
            calls.append(name)
            return original(data)

        monkeypatch.setattr(algorithm, f"_sequential_{name}", recording)

    expected = [
        ["critic_two", "critic_one"],
        ["critic_two", "critic_one"],
        ["critic_two", "critic_one"],
        ["critic_two", "critic_one", "actor"],
    ]
    previous = 0
    for step, expected_calls in enumerate(expected, start=1):
        metrics = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
        assert calls[previous:] == expected_calls
        previous = len(calls)
        assert metrics["critic_two_update_applied"] == 1.0
        assert metrics["critic_two_update_steps"] == step
        assert metrics["actor_update_applied"] == float(step == 4)


def test_board_identical_initialization_copies_values_not_parameters():
    algorithm = BoardSharedDDDPG(board_settings(critic_initialization="identical"))
    one_before = copies(algorithm.critic_one)
    two_before = copies(algorithm.critic_two)
    for one, two in zip(one_before, two_before, strict=True):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert all(
        one.data_ptr() != two.data_ptr()
        for one, two in zip(algorithm.critic_one.parameters(), algorithm.critic_two.parameters(), strict=True)
    )

    algorithm._sequential_critic_two(batch())
    unchanged(one_before, algorithm.critic_one)
    assert any(
        not torch.equal(one, two)
        for one, two in zip(algorithm.critic_one.parameters(), algorithm.critic_two.parameters(), strict=True)
    )


def test_board_registry_and_manifest_preserve_equations_and_isolate_two_hypotheses():
    config = load_config(algorithm="three_network_shared_d_board")
    algorithm = create_algorithm(
        "three_network_shared_d_board",
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    assert isinstance(algorithm, BoardSharedDDDPG)
    assert algorithm.settings.critic_objective == "shared_error"
    assert algorithm.settings.update_schedule == "sequential"
    assert algorithm.settings.actor_state_source == "next"
    assert algorithm.settings.alpha == algorithm.settings.alpha_prime == pytest.approx(0.1)
    assert not hasattr(algorithm, "target_actor")
    assert not hasattr(algorithm, "target_critic")

    repository = Path(__file__).resolve().parents[2]
    path = repository / "experiments/manifests/pendulum_shared_d_board_20k_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert len(planned) == 5
    assert manifest.output_root == Path("runs/pendulum-shared-d-board-20k-01")
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert set(manifest.training_seeds).isdisjoint(manifest.evaluation_seeds)
    assert all(run.config.experiment.total_steps == 20_000 for run in planned)
    assert {run.variant_id for run in planned} == {
        "ddpg",
        "board-reference",
        "board-actor-warmup",
        "board-identical-init",
        "board-warmup-identical",
    }
    proposed = [run for run in planned if run.config.algorithm.name == "three_network_shared_d_board"]
    assert len(proposed) == 4
    assert {run.config.algorithm.actor_learning_starts for run in proposed} == {0, 5000}
    assert {run.config.algorithm.critic_initialization for run in proposed} == {
        "independent",
        "identical",
    }
    for run in proposed:
        settings = run.config.algorithm
        assert settings.critic_one_learning_rate == settings.critic_two_learning_rate == pytest.approx(3e-4)
        assert settings.actor_update_delay == 1
        assert settings.alpha == settings.alpha_prime == pytest.approx(0.1)
        assert settings.beta_regularizer.mode == "none"
        assert "target_networks" not in settings

    confirmatory = load_manifest(repository / "experiments/manifests/confirmatory.yaml")
    assert set(manifest.training_seeds).isdisjoint(confirmatory.training_seeds)


def test_synchronous_critics_match_one_snapshot_reference_and_do_not_update_actor(monkeypatch):
    algorithm = SynchronousSharedDDDPG(synchronous_settings(alpha=0.3, alpha_prime=0.4))
    data = batch()
    actor_before = copies(algorithm.actor)
    next_actions = algorithm.actor(data.next_observations).detach()
    y = algorithm.critic_one(data.observations, data.actions)
    y_next = algorithm.critic_one(data.next_observations, next_actions)
    z_current = algorithm.critic_two(data.observations, data.actions)
    z_next = algorithm.critic_two(data.next_observations, next_actions)
    d_for_two = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y.detach()
    d_for_one = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next.detach() - y
    q2_reference = (
        0.5 * d_for_two.square().mean()
        + 0.5 * algorithm.settings.alpha_prime * (z_current - y.detach()).square().mean()
    )
    q1_reference = (
        0.5 * d_for_one.square().mean()
        + 0.5 * algorithm.settings.alpha * (y_next - z_next.detach()).square().mean()
    )
    expected_two = torch.autograd.grad(q2_reference, tuple(algorithm.critic_two.parameters()))
    expected_one = torch.autograd.grad(q1_reference, tuple(algorithm.critic_one.parameters()))
    captured = {}

    def capture_two():
        captured["two"] = tuple(
            parameter.grad.detach().clone() for parameter in algorithm.critic_two.parameters()
        )

    def capture_one():
        captured["one"] = tuple(
            parameter.grad.detach().clone() for parameter in algorithm.critic_one.parameters()
        )

    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", capture_two)
    monkeypatch.setattr(algorithm.critic_one_optimizer, "step", capture_one)

    q1_result, q2_result, actor_result = algorithm._simultaneous_update(data)

    assert actor_result is None
    assert q1_result.loss == pytest.approx(float(q1_reference.detach()))
    assert q2_result.loss == pytest.approx(float(q2_reference.detach()))
    for expected, actual in zip(expected_one, captured["one"], strict=True):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(expected_two, captured["two"], strict=True):
        torch.testing.assert_close(actual, expected)
    unchanged(actor_before, algorithm.actor)


def test_synchronous_update_commits_both_critics_before_delayed_actor_and_resumes():
    configured = synchronous_settings(actor_learning_starts=2, actor_update_delay=2)
    algorithm = SynchronousSharedDDDPG(configured)
    actor_before = copies(algorithm.actor)
    critic_one_before = copies(algorithm.critic_one)
    critic_two_before = copies(algorithm.critic_two)

    for expected_step in range(1, 4):
        metrics = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
        assert metrics["update_steps"] == expected_step
        assert metrics["critic_snapshot_synchronous"] == 1
        assert metrics["critic_two_update_applied"] == 1
        assert metrics["actor_update_applied"] == 0
    unchanged(actor_before, algorithm.actor)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_one_before, algorithm.critic_one.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_two_before, algorithm.critic_two.parameters(), strict=True)
    )

    resumed = SynchronousSharedDDDPG(configured)
    resumed.load_state_dict(deepcopy(algorithm.state_dict()))
    original = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
    resumed_result = {record.name: record.value for record in resumed.update_from_batch(batch()).metrics}
    assert original == resumed_result
    assert original["actor_update_applied"] == 1
    assert original["actor_update_steps"] == 1
    for name in ("actor", "critic_one", "critic_two"):
        unchanged(copies(getattr(algorithm, name)), getattr(resumed, name))


def test_action_utilization_diagnostics_are_normalized_monitoring_only():
    algorithm = SynchronousSharedDDDPG(synchronous_settings(diagnostic_frequency=1))
    actor_before = copies(algorithm.actor)
    metrics = {record.name: record.value for record in algorithm.update_from_batch(batch()).metrics}
    for name in (
        "actor_current_action_utilization_mean",
        "actor_current_action_utilization_max",
        "actor_next_action_utilization_mean",
        "actor_next_action_utilization_max",
    ):
        assert 0 <= metrics[name] <= 1
    unchanged(actor_before, algorithm.actor)


@pytest.mark.parametrize("source", ["current", "next"])
def test_synchronous_actor_source_is_explicit_and_gradients_pass_through_fixed_q2(monkeypatch, source):
    algorithm = SynchronousSharedDDDPG(synchronous_settings(actor_state_source=source))
    data = batch()
    critic_one_before = copies(algorithm.critic_one)
    critic_two_before = copies(algorithm.critic_two)
    states = data.observations if source == "current" else data.next_observations
    critic_parameters = (*algorithm.critic_one.parameters(), *algorithm.critic_two.parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    reference = -algorithm.critic_two(states, algorithm.actor(states)).mean()
    expected = torch.autograd.grad(reference, tuple(algorithm.actor.parameters()))
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    monkeypatch.setattr(algorithm.actor_optimizer, "step", lambda: None)

    algorithm._sequential_actor(data)

    assert any(torch.count_nonzero(gradient).item() for gradient in expected)
    for gradient, parameter in zip(expected, algorithm.actor.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    unchanged(critic_one_before, algorithm.critic_one)
    unchanged(critic_two_before, algorithm.critic_two)


def test_synchronous_registry_has_no_targets_and_accepts_both_actor_sources():
    for source in ("current", "next"):
        config = load_config(
            algorithm="three_network_shared_d_synchronous",
            overrides=[f"algorithm.actor_state_source={source}"],
        )
        algorithm = create_algorithm(
            "three_network_shared_d_synchronous",
            algorithm_config=config.algorithm,
            observation_dim=3,
            action_low=np.array([-2.0]),
            action_high=np.array([2.0]),
        )
        assert isinstance(algorithm, SynchronousSharedDDDPG)
        assert algorithm.settings.update_schedule == "simultaneous"
        assert algorithm.settings.actor_state_source == source
        assert algorithm.settings.actor_learning_starts == 5_000
        assert algorithm.settings.actor_update_delay == 2
        assert "target_networks" not in config.algorithm
        assert not hasattr(algorithm, "target_actor")
        assert not hasattr(algorithm, "target_critic")


@pytest.mark.parametrize(
    "override",
    [
        "algorithm.update_schedule=sequential",
        "algorithm.actor_learning_starts=-1",
        "algorithm.actor_update_delay=0",
        "algorithm.critic_objective=separate_td",
        "algorithm.agreement_mode=none",
        "algorithm.beta_regularizer.mode=l2_y_z",
    ],
)
def test_synchronous_incompatible_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(algorithm="three_network_shared_d_synchronous", overrides=[override])


def test_stabilized_15k_manifest_is_paired_isolated_and_changes_one_factor_per_structure():
    repository = Path(__file__).resolve().parents[2]
    path = repository / "experiments/manifests/pendulum_shared_d_stabilized_15k_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert len(planned) == 4
    assert manifest.output_root == Path("runs/pendulum-shared-d-stabilized-15k-01")
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert set(manifest.training_seeds).isdisjoint(manifest.evaluation_seeds)
    assert all(run.config.experiment.total_steps == 15_000 for run in planned)
    assert sum(run.config.experiment.total_steps for run in planned) == 60_000
    assert {run.variant_id for run in planned} == {
        "ddpg",
        "sequential-conservative",
        "synchronous-next",
        "synchronous-current",
    }
    proposed = [run for run in planned if run.variant_id != "ddpg"]
    assert {run.config.algorithm.name for run in proposed} == {
        "three_network_shared_d_board",
        "three_network_shared_d_synchronous",
    }
    for run in proposed:
        configured = run.config.algorithm
        assert configured.critic_objective == "shared_error"
        assert configured.agreement_mode == "both"
        assert configured.alpha == configured.alpha_prime == pytest.approx(0.1)
        assert configured.beta_regularizer.mode == "none"
        assert configured.actor_learning_starts == 5_000
        assert configured.actor_update_delay == 2
        assert configured.actor_learning_rate == pytest.approx(5e-5)
        assert configured.critic_one_learning_rate == pytest.approx(1e-4)
        assert configured.critic_two_learning_rate == pytest.approx(1e-4)
        assert configured.gradient_clip_norm == pytest.approx(5.0)
        assert configured.critic_initialization == "independent"
        assert "target_networks" not in configured
    by_id = {run.variant_id: run for run in planned}
    assert by_id["sequential-conservative"].config.algorithm.update_schedule == "sequential"
    assert by_id["synchronous-next"].config.algorithm.update_schedule == "simultaneous"
    assert by_id["synchronous-current"].config.algorithm.update_schedule == "simultaneous"
    assert by_id["sequential-conservative"].config.algorithm.actor_state_source == "next"
    assert by_id["synchronous-next"].config.algorithm.actor_state_source == "next"
    assert by_id["synchronous-current"].config.algorithm.actor_state_source == "current"
    for key in (
        "total_steps",
        "warmup_steps",
        "batch_size",
        "replay_capacity",
        "evaluation_frequency",
        "evaluation_episodes",
        "bootstrap_on_truncation",
        "clip_actions",
        "tracking",
    ):
        assert len({str(run.config.experiment[key]) for run in planned}) == 1
    assert len({run.run_id for run in planned}) == len(planned)
    assert [run.run_id for run in planned] == [run.run_id for run in expand_manifest(load_manifest(path))]
    confirmatory = load_manifest(repository / "experiments/manifests/confirmatory.yaml")
    assert set(manifest.training_seeds).isdisjoint(confirmatory.training_seeds)
