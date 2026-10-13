"""Board-row correspondence, independent gradients, and slow-Actor clocks.

All batches and replay transitions are synthetic. No environment is constructed.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from three_network_ddpg.algorithms.paired_shared_d import (
    ConditionedPairedSharedDDDPG,
    ConditionedPairedSharedDSettings,
    PairedSharedDDDPG,
    PairedSharedDSettings,
)
from three_network_ddpg.algorithms.registry import create_algorithm
from three_network_ddpg.algorithms.shared_d import BoardSharedDDDPG, BoardSharedDSettings
from three_network_ddpg.config import load_config
from three_network_ddpg.core.batches import Transition, TransitionBatch
from three_network_ddpg.core.replay import ReplayBuffer
from three_network_ddpg.protocol.manifest import expand_manifest, load_manifest


def settings(**changes):
    values = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
        "seed": 47,
        "diagnostic_frequency": None,
    }
    return PairedSharedDSettings(**(values | changes))


def batch():
    generator = torch.Generator().manual_seed(9)
    terminated = torch.zeros((7, 1), dtype=torch.bool)
    terminated[-1] = True
    truncated = torch.zeros_like(terminated)
    truncated[-2] = True
    return TransitionBatch(
        observations=torch.randn((7, 3), generator=generator),
        actions=torch.randn((7, 1), generator=generator),
        rewards=torch.randn((7, 1), generator=generator),
        next_observations=torch.randn((7, 3), generator=generator) + 2,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=(~terminated).float(),
    )


def copies(module):
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def unchanged(before, module):
    for original, parameter in zip(before, module.parameters(), strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


@pytest.mark.parametrize("bootstrap_on_truncation", [True, False])
def test_replay_ring_samples_all_columns_with_the_same_transition_indices(bootstrap_on_truncation):
    replay = ReplayBuffer(
        capacity=13,
        observation_shape=(3,),
        action_shape=(1,),
        seed=11,
        bootstrap_on_truncation=bootstrap_on_truncation,
        device="cpu",
    )
    for i in range(20):
        replay.add(
            Transition(
                np.array([i, i + 1, i + 2]),
                np.array([i + 0.25]),
                -float(i),
                np.array([i + 100, i + 101, i + 102]),
                i % 3 == 0,
                i % 4 == 0,
            )
        )
    for _ in range(3):
        data = replay.sample(8)
        ids = data.observations[:, 0]
        assert ids.min() >= 7  # The first seven transitions were overwritten.
        torch.testing.assert_close(data.actions[:, 0], ids + 0.25)
        torch.testing.assert_close(data.rewards[:, 0], -ids)
        torch.testing.assert_close(data.next_observations, data.observations + 100)
        assert torch.equal(data.terminated[:, 0], ids.remainder(3) == 0)
        assert torch.equal(data.truncated[:, 0], ids.remainder(4) == 0)
        expected_mask = ~data.terminated
        if not bootstrap_on_truncation:
            expected_mask &= ~data.truncated
        torch.testing.assert_close(data.bootstrap_mask, expected_mask.float())


def test_four_board_vectors_reuse_exact_ordered_current_and_next_input_pairs():
    algorithm, data = PairedSharedDDDPG(settings()), batch()
    actor_calls, one_calls, two_calls = [], [], []
    handles = [
        algorithm.actor.register_forward_pre_hook(lambda module, args: actor_calls.append(args)),
        algorithm.critic_one.register_forward_pre_hook(lambda module, args: one_calls.append(args)),
        algorithm.critic_two.register_forward_pre_hook(lambda module, args: two_calls.append(args)),
    ]
    try:
        algorithm._sequential_update(data)
    finally:
        for handle in handles:
            handle.remove()
    assert len(actor_calls) == 1  # One detached a_next for BOTH critic phases.
    assert actor_calls[0][0] is data.next_observations
    assert len(one_calls) == 3 and len(two_calls) == 3
    for observations, actions in (one_calls[0], one_calls[1], two_calls[0]):
        assert observations is data.observations
        assert actions is data.actions  # Replay actions, NOT actor(s).
    next_pairs = (two_calls[1], two_calls[2], one_calls[2])
    for observations, actions in next_pairs:
        assert observations is data.next_observations
        assert actions is next_pairs[0][1]  # Same tensor, no resampling/permutation.
        assert not actions.requires_grad
    with torch.no_grad():
        torch.testing.assert_close(next_pairs[0][1], algorithm.actor(data.next_observations))


@pytest.mark.parametrize("phase", ["one", "two"])
def test_paired_losses_and_gradients_match_independent_board_reference(monkeypatch, phase):
    algorithm, data = PairedSharedDDDPG(settings(alpha=0.3, alpha_prime=0.4)), batch()
    a_next = algorithm.actor(data.next_observations).detach()
    if phase == "two":
        y = algorithm.critic_one(data.observations, data.actions).detach()
        z_next = algorithm.critic_two(data.next_observations, a_next)
        z_current = algorithm.critic_two(data.observations, data.actions)
        d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
        reference = (
            0.5 * d.square().mean() + 0.5 * algorithm.settings.alpha_prime * (z_current - y).square().mean()
        )
        own, other = algorithm.critic_two, algorithm.critic_one
    else:
        z_next = algorithm.critic_two(data.next_observations, a_next).detach()
        y = algorithm.critic_one(data.observations, data.actions)
        y_next = algorithm.critic_one(data.next_observations, a_next)
        d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
        reference = (
            0.5 * d.square().mean() + 0.5 * algorithm.settings.alpha * (y_next - z_next).square().mean()
        )
        own, other = algorithm.critic_one, algorithm.critic_two
    expected = torch.autograd.grad(reference, tuple(own.parameters()))
    actor_before, other_before = copies(algorithm.actor), copies(other)
    monkeypatch.setattr(getattr(algorithm, f"critic_{phase}_optimizer"), "step", lambda: None)
    result = getattr(algorithm, f"_paired_critic_{phase}")(algorithm._paired_inputs(data))
    assert result.loss == pytest.approx(float(reference.detach()))
    for actual, gradient in zip(own.parameters(), expected, strict=True):
        torch.testing.assert_close(actual.grad, gradient)
    unchanged(actor_before, algorithm.actor)
    unchanged(other_before, other)
    assert all(parameter.grad is None for parameter in (*algorithm.actor.parameters(), *other.parameters()))


@pytest.mark.parametrize("phase", ["one", "two"])
def test_paired_critic_step_changes_only_its_own_network(phase):
    algorithm = PairedSharedDDDPG(settings())
    own = getattr(algorithm, f"critic_{phase}")
    other = algorithm.critic_one if phase == "two" else algorithm.critic_two
    before, other_before, actor_before = copies(own), copies(other), copies(algorithm.actor)
    getattr(algorithm, f"_paired_critic_{phase}")(algorithm._paired_inputs(batch()))
    assert any(not torch.equal(a, b) for a, b in zip(before, own.parameters(), strict=True))
    unchanged(other_before, other)
    unchanged(actor_before, algorithm.actor)


def test_q2_shared_d_has_next_input_gradient_even_with_zero_agreement(monkeypatch):
    algorithm, data = PairedSharedDDDPG(settings(alpha=0, alpha_prime=0)), batch()
    inputs = algorithm._paired_inputs(data)
    y = algorithm.critic_one(data.observations, data.actions).detach()
    z_next = algorithm.critic_two(data.next_observations, inputs.next_actions)
    z_current = algorithm.critic_two(data.observations, data.actions)
    d = data.rewards + algorithm.settings.gamma * data.bootstrap_mask * z_next - y
    loss = 0.5 * d.square().mean()
    next_grad, current_grad = torch.autograd.grad(
        loss, (z_next, z_current), allow_unused=True, retain_graph=True
    )
    torch.testing.assert_close(
        next_grad, algorithm.settings.gamma * data.bootstrap_mask * d / data.batch_size
    )
    assert current_grad is None and next_grad[-1] == 0 and next_grad[-2] != 0
    expected = torch.autograd.grad(loss, tuple(algorithm.critic_two.parameters()))
    monkeypatch.setattr(algorithm.critic_two_optimizer, "step", lambda: None)
    algorithm._paired_critic_two(inputs)
    for parameter, gradient in zip(algorithm.critic_two.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_q1_uses_fresh_q2_values_and_keeps_its_next_agreement_gradient(monkeypatch):
    algorithm, data = PairedSharedDDDPG(settings(alpha=0.5)), batch()
    inputs = algorithm._paired_inputs(data)
    with torch.no_grad():
        stale = algorithm.critic_two(data.next_observations, inputs.next_actions).clone()
    algorithm._paired_critic_two(inputs)
    with torch.no_grad():
        fresh = algorithm.critic_two(data.next_observations, inputs.next_actions)
    assert not torch.equal(stale, fresh)
    y = algorithm.critic_one(data.observations, data.actions)
    y_next = algorithm.critic_one(data.next_observations, inputs.next_actions)
    td = 0.5 * (data.rewards + algorithm.settings.gamma * data.bootstrap_mask * fresh - y).square().mean()
    reference = td + 0.5 * algorithm.settings.alpha * (y_next - fresh).square().mean()
    parameters = tuple(algorithm.critic_one.parameters())
    expected = torch.autograd.grad(reference, parameters, retain_graph=True)
    td_only = torch.autograd.grad(td, parameters)
    assert any(not torch.allclose(a, b) for a, b in zip(expected, td_only, strict=True))
    monkeypatch.setattr(algorithm.critic_one_optimizer, "step", lambda: None)
    algorithm._paired_critic_one(inputs)
    for parameter, gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_actor_gradient_passes_through_next_action_while_critics_stay_fixed(monkeypatch):
    algorithm, data = PairedSharedDDDPG(settings()), batch()
    before_one, before_two = copies(algorithm.critic_one), copies(algorithm.critic_two)
    critics = (*algorithm.critic_one.parameters(), *algorithm.critic_two.parameters())
    for parameter in critics:
        parameter.requires_grad_(False)
    reference = -algorithm.critic_two(data.next_observations, algorithm.actor(data.next_observations)).mean()
    expected = torch.autograd.grad(reference, tuple(algorithm.actor.parameters()))
    for parameter in critics:
        parameter.requires_grad_(True)
    assert any(torch.count_nonzero(gradient).item() for gradient in expected)
    monkeypatch.setattr(algorithm.actor_optimizer, "step", lambda: None)
    algorithm._sequential_actor(data)
    for parameter, gradient in zip(algorithm.actor.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    unchanged(before_one, algorithm.critic_one)
    unchanged(before_two, algorithm.critic_two)
    assert all(parameter.grad is None for parameter in critics)


@pytest.mark.parametrize("delay", [5, 10])
@pytest.mark.parametrize("start", [0, 3])
def test_both_critics_step_every_iteration_actor_clock_and_resume_are_exact(delay, start):
    algorithm = PairedSharedDDDPG(settings(actor_update_delay=delay, actor_learning_starts=start))
    data = batch()
    for step in range(1, 21):
        actor_before = copies(algorithm.actor)
        metrics = {record.name: record.value for record in algorithm.update_from_batch(data).metrics}
        due = step > start and (step - start) % delay == 0
        assert metrics["actor_update_applied"] == float(due)
        assert metrics["actor_update_steps"] == max(0, step - start) // delay
        assert metrics["critic_two_update_steps"] == step
        assert metrics["critic_pair_update_steps"] == step
        assert metrics["paired_batch_size"] == data.batch_size
        if not due:
            unchanged(actor_before, algorithm.actor)
            assert "actor_gradient_norm" not in metrics
        if step == 9:  # Restore just before an Actor boundary, without resetting the clock.
            restored = PairedSharedDDDPG(algorithm.settings)
            restored.load_state_dict(deepcopy(algorithm.state_dict()))
        elif step > 9:
            restored_metrics = {r.name: r.value for r in restored.update_from_batch(data).metrics}
            assert restored_metrics == pytest.approx(metrics)
    for name in ("actor", "critic_one", "critic_two"):
        unchanged(copies(getattr(algorithm, name)), getattr(restored, name))
    for name, count in (("critic_one", 20), ("critic_two", 20), ("actor", (20 - start) // delay)):
        optimizer = getattr(algorithm, f"{name}_optimizer")
        assert all(int(state["step"]) == count for state in optimizer.state.values())


def test_phase_order_is_q2_fresh_q1_then_due_actor(monkeypatch):
    algorithm, data = PairedSharedDDDPG(settings()), batch()
    algorithm._update_step = 9
    events = []
    for name in ("critic_two", "critic_one", "actor"):
        optimizer = getattr(algorithm, f"{name}_optimizer")
        original = optimizer.step

        def record(*args, _name=name, _original=original, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(optimizer, "step", record)
    algorithm.update_from_batch(data)
    assert events == ["critic_two", "critic_one", "actor"]


def test_paired_and_historical_board_updates_agree_when_all_settings_match():
    paired = PairedSharedDDDPG(settings())
    old_settings = paired.settings.to_dict()
    old_settings.pop("paired_batch_contract")
    board = BoardSharedDDDPG(BoardSharedDSettings(**old_settings))
    for _ in range(11):
        paired.update_from_batch(batch())
        board.update_from_batch(batch())
    for name in ("actor", "critic_one", "critic_two"):
        unchanged(copies(getattr(paired, name)), getattr(board, name))


@pytest.mark.parametrize("changed", ["contract", "delay", "start"])
def test_resume_rejects_incompatible_schedule_or_historical_checkpoint(changed):
    algorithm = PairedSharedDDDPG(settings())
    state = deepcopy(algorithm.state_dict())
    if changed == "contract":
        state["settings"].pop("paired_batch_contract")
    elif changed == "delay":
        state["settings"]["actor_update_delay"] = 5
    else:
        state["settings"]["actor_learning_starts"] = 100
    before = copies(algorithm.actor)
    with pytest.raises(ValueError, match="settings"):
        algorithm.load_state_dict(state)
    unchanged(before, algorithm.actor)


@pytest.mark.parametrize(
    "override",
    [
        "algorithm.actor_update_delay=1",
        "algorithm.actor_update_delay=5.5",
        "algorithm.actor_learning_starts=-1",
        "algorithm.paired_batch_contract=unpaired",
        "algorithm.critic_objective=separate_td",
        "algorithm.actor_state_source=current",
        "algorithm.update_schedule=simultaneous",
        "algorithm.critic_initialization=identical",
        "algorithm.target_networks.actor=true",
        "algorithm.beta_regularizer.mode=l2_y_z",
    ],
)
def test_incompatible_paired_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(algorithm="three_network_shared_d_paired", overrides=[override])


def test_registry_has_three_independent_networks_no_targets_and_monitoring_only_validation():
    config = load_config(
        algorithm="three_network_shared_d_paired", overrides=["algorithm.hidden_layers=[8,8]"]
    )
    algorithm = create_algorithm(
        config.algorithm.name,
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    assert isinstance(algorithm, PairedSharedDDDPG)
    assert not algorithm.requires_validation_batch and "target_networks" not in config.algorithm
    ids = [
        {id(parameter) for parameter in getattr(algorithm, name).parameters()}
        for name in ("actor", "critic_one", "critic_two")
    ]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert not hasattr(algorithm, "target_actor") and not hasattr(algorithm, "target_critic")
    reference = PairedSharedDDDPG(algorithm.settings)
    validation = batch()
    validation.rewards.fill_(10000)
    algorithm.update_from_batch(batch(), validation_batch=validation)
    reference.update_from_batch(batch())
    for name in ("actor", "critic_one", "critic_two"):
        unchanged(copies(getattr(algorithm, name)), getattr(reference, name))


def test_manual_manifest_has_stable_unique_ids_paired_seeds_and_only_declared_differences():
    path = Path(__file__).resolve().parents[2] / "experiments/manifests/pendulum_shared_d_paired_15k_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert planned == expand_manifest(manifest)
    assert len(planned) == len({run.run_id for run in planned}) == 3
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert not set(manifest.training_seeds) & set(manifest.evaluation_seeds)
    assert manifest.workers == 1
    assert manifest.output_root == Path("runs/pendulum-shared-d-paired-15k-01")
    assert sum(run.config.experiment.total_steps for run in planned) == 45000
    assert {run.variant_id for run in planned} == {"ddpg", "paired-10-to-1", "paired-10-to-2"}
    first, second = planned[1:]
    first_config, second_config = OmegaConf.to_container(first.config), OmegaConf.to_container(second.config)
    assert first_config["algorithm"].pop("actor_update_delay") == 10
    assert second_config["algorithm"].pop("actor_update_delay") == 5
    assert first_config == second_config
    for run in planned:
        assert run.training_seed == 11
        assert run.evaluation_seeds == manifest.evaluation_seeds
        assert run.config.environment == planned[0].config.environment
        assert run.config.experiment.total_steps == 15000
        assert run.config.experiment.evaluation_frequency == 500
        assert run.config.experiment.evaluation_episodes == 5
        assert run.config.experiment.warmup_steps == 1000
        assert run.config.experiment.batch_size == 64
        assert run.config.experiment.replay_capacity == 100000
    for run in (first, second):
        cfg = run.config.algorithm
        assert cfg.name == "three_network_shared_d_paired"
        assert cfg.actor_learning_starts == 0
        assert cfg.actor_state_source == "next"
        assert cfg.alpha == cfg.alpha_prime == 0.1
        assert cfg.critic_objective == "shared_error" and cfg.agreement_mode == "both"
        assert cfg.paired_batch_contract == "board_aligned_v1"
        assert cfg.beta_regularizer.mode == "none" and cfg.beta_regularizer.coefficient == 0
        assert "target_networks" not in cfg and not cfg.guard.validation_batch


def test_conditioned_paired_name_preserves_paired_update_math_with_an_explicit_contract():
    config = load_config(
        algorithm="three_network_shared_d_conditioned_paired",
        overrides=["algorithm.hidden_layers=[8,8]", "algorithm.actor_update_delay=2"],
    )
    registered = create_algorithm(
        config.algorithm.name,
        algorithm_config=config.algorithm,
        observation_dim=3,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    assert isinstance(registered, ConditionedPairedSharedDDDPG)
    assert not registered.requires_validation_batch
    assert not hasattr(registered, "target_actor") and not hasattr(registered, "target_critic")

    conditioned_settings = ConditionedPairedSharedDSettings(
        observation_dim=3,
        action_dim=1,
        action_low=(-2.0,),
        action_high=(2.0,),
        hidden_layers=(8, 8),
        seed=71,
        diagnostic_frequency=None,
        actor_update_delay=2,
    )
    values = conditioned_settings.to_dict()
    assert values.pop("conditioning_contract") == "asymmetric_current_anchor_v1"
    reference_settings = PairedSharedDSettings(**values)
    conditioned = ConditionedPairedSharedDDDPG(conditioned_settings)
    reference = PairedSharedDDDPG(reference_settings)

    conditioned.update_from_batch(batch())
    reference.update_from_batch(batch())

    for name in ("actor", "critic_one", "critic_two"):
        for actual, expected in zip(
            getattr(conditioned, name).parameters(),
            getattr(reference, name).parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert conditioned.settings.alpha == 0.01
    assert conditioned.settings.alpha_prime == 10.0
    assert conditioned.settings.actor_learning_starts == 5_000


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"alpha": 0.1}, "alpha=0.01"),
        ({"alpha_prime": 0.1}, "alpha_prime=10.0"),
        ({"actor_learning_starts": 0}, "actor_learning_starts=5000"),
        ({"conditioning_contract": "wrong"}, "conditioning_contract"),
    ],
)
def test_conditioned_paired_contract_rejects_accidental_semantic_drift(change, message):
    values = {
        "observation_dim": 3,
        "action_dim": 1,
        "action_low": (-2.0,),
        "action_high": (2.0,),
        "hidden_layers": (8, 8),
    }
    with pytest.raises(ValueError, match=message):
        ConditionedPairedSharedDSettings(**(values | change))


def test_conditioned_paired_20k_manifest_is_stable_paired_and_changes_only_actor_clock():
    repository = Path(__file__).resolve().parents[2]
    path = repository / "experiments/manifests/pendulum_shared_d_conditioned_paired_20k_01.yaml"
    manifest = load_manifest(path)
    planned = expand_manifest(manifest)
    assert planned == expand_manifest(manifest)
    assert len(planned) == len({run.run_id for run in planned}) == 4
    assert manifest.training_seeds == (11,)
    assert manifest.evaluation_seeds == (1001, 1002, 1003, 1004, 1005)
    assert set(manifest.training_seeds).isdisjoint(manifest.evaluation_seeds)
    confirmatory = load_manifest(repository / "experiments/manifests/confirmatory.yaml")
    assert set(manifest.training_seeds).isdisjoint(confirmatory.training_seeds)
    assert manifest.workers == 1
    assert manifest.output_root == Path("runs/pendulum-shared-d-conditioned-paired-20k-01")
    assert sum(run.config.experiment.total_steps for run in planned) == 80_000
    assert {run.variant_id for run in planned} == {
        "ddpg",
        "conditioned-paired-2-to-1",
        "conditioned-paired-10-to-2",
        "conditioned-paired-10-to-1",
    }

    proposed = planned[1:]
    normalized = []
    for run, expected_delay in zip(proposed, (2, 5, 10), strict=True):
        assert run.training_seed == 11 and run.evaluation_seeds == manifest.evaluation_seeds
        assert run.config.environment == planned[0].config.environment
        assert run.config.experiment.total_steps == 20_000
        assert run.config.experiment.warmup_steps == 1_000
        assert run.config.experiment.batch_size == 64
        assert run.config.experiment.replay_capacity == 100_000
        assert run.config.experiment.evaluation_frequency == 500
        assert run.config.experiment.evaluation_episodes == 5
        cfg = run.config.algorithm
        assert cfg.name == "three_network_shared_d_conditioned_paired"
        assert cfg.alpha == 0.01 and cfg.alpha_prime == 10.0
        assert cfg.actor_learning_starts == 5_000 and cfg.actor_update_delay == expected_delay
        assert cfg.critic_objective == "shared_error" and cfg.agreement_mode == "both"
        assert cfg.paired_batch_contract == "board_aligned_v1"
        assert cfg.conditioning_contract == "asymmetric_current_anchor_v1"
        assert cfg.actor_state_source == "next" and cfg.update_schedule == "sequential"
        assert cfg.critic_initialization == "independent"
        assert cfg.beta_regularizer.mode == "none" and cfg.beta_regularizer.coefficient == 0
        assert "target_networks" not in cfg and not cfg.guard.validation_batch
        values = OmegaConf.to_container(run.config)
        values["algorithm"].pop("actor_update_delay")
        normalized.append(values)
    assert normalized[0] == normalized[1] == normalized[2]
