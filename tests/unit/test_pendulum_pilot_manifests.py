"""Static contracts for the manually launched Pendulum Pilot 01."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from three_network_ddpg.protocol.manifest import (
    ExperimentManifest,
    PlannedRun,
    expand_manifest,
    load_manifest,
)

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST_DIRECTORY = REPOSITORY / "experiments" / "manifests"
PLAN_PATH = REPOSITORY / "experiments" / "plans" / "pendulum_pilot_01.yaml"
CONFIRMATORY_PATH = MANIFEST_DIRECTORY / "confirmatory.yaml"
DEVELOPMENT_TRAINING_SEEDS = (11, 22, 33)
EVALUATION_SEEDS = (1001, 1002, 1003, 1004, 1005)

STAGES = {
    "a": {
        "path": MANIFEST_DIRECTORY / "pendulum_pilot_01_stage_a.yaml",
        "name": "pendulum-pilot-01-stage-a",
        "variants": (
            "ddpg-pipeline-smoke",
            "ddpg-no-target-actor-pipeline-smoke",
            "three-network-shared-error-pipeline-smoke",
        ),
        "seeds": (11,),
        "runs": 3,
        "steps": 2000,
    },
    "b": {
        "path": MANIFEST_DIRECTORY / "pendulum_pilot_01_stage_b.yaml",
        "name": "pendulum-pilot-01-stage-b",
        "variants": ("ddpg", "ddpg-no-target-actor", "three-network-shared-error"),
        "seeds": DEVELOPMENT_TRAINING_SEEDS,
        "runs": 9,
        "steps": 20000,
    },
    "c": {
        "path": MANIFEST_DIRECTORY / "pendulum_pilot_01_stage_c.yaml",
        "name": "pendulum-pilot-01-stage-c",
        "variants": (
            "three-network-none",
            "three-network-current",
            "three-network-next",
            "three-network-both",
        ),
        "seeds": DEVELOPMENT_TRAINING_SEEDS,
        "runs": 12,
        "steps": 20000,
    },
}


def _manifest(stage: str) -> ExperimentManifest:
    return load_manifest(STAGES[stage]["path"])


def _seed_run(planned: tuple[PlannedRun, ...], variant_id: str) -> PlannedRun:
    return next(run for run in planned if run.variant_id == variant_id and run.training_seed == 11)


@pytest.mark.parametrize("stage", ("a", "b", "c"))
def test_pilot_manifests_parse_expand_and_pair_development_seeds(stage: str) -> None:
    expected = STAGES[stage]
    manifest = _manifest(stage)
    first = expand_manifest(manifest)
    second = expand_manifest(manifest)
    assert manifest.name == expected["name"]
    assert manifest.training_seeds == expected["seeds"]
    assert manifest.evaluation_seeds == EVALUATION_SEEDS
    assert manifest.output_root == Path("runs/pendulum-pilot-01")
    assert manifest.workers == 1
    assert len(first) == expected["runs"]
    assert [run.run_id for run in first] == [run.run_id for run in second]
    assert len({run.run_id for run in first}) == len(first)
    assert tuple(variant.variant_id for variant in manifest.variants) == expected["variants"]
    assert {run.training_seed for run in first} == set(expected["seeds"])
    assert all(run.evaluation_seeds == EVALUATION_SEEDS for run in first)
    for variant_id in expected["variants"]:
        assert tuple(run.training_seed for run in first if run.variant_id == variant_id) == expected["seeds"]
        assert all(
            int(run.config.experiment.total_steps) == expected["steps"]
            for run in first
            if run.variant_id == variant_id
        )


def test_pilot_seeds_are_disjoint_from_evaluation_and_confirmatory_training() -> None:
    confirmatory = load_manifest(CONFIRMATORY_PATH)
    for stage in STAGES:
        manifest = _manifest(stage)
        assert not set(manifest.training_seeds) & set(manifest.evaluation_seeds)
        assert not set(manifest.training_seeds) & set(confirmatory.training_seeds)
        assert not set(manifest.evaluation_seeds) & set(confirmatory.training_seeds)
    assert set(DEVELOPMENT_TRAINING_SEEDS).isdisjoint(confirmatory.training_seeds)


def test_stage_b_has_only_the_required_baselines_and_target_free_reference() -> None:
    planned = expand_manifest(_manifest("b"))
    ddpg = _seed_run(planned, "ddpg")
    no_target_actor = _seed_run(planned, "ddpg-no-target-actor")
    reference = _seed_run(planned, "three-network-shared-error")
    assert ddpg.config.algorithm.name == "ddpg"
    assert ddpg.config.algorithm.target_networks.actor
    assert ddpg.config.algorithm.target_networks.critic
    assert no_target_actor.config.algorithm.name == "ddpg_no_target_actor"
    assert not no_target_actor.config.algorithm.target_networks.actor
    assert no_target_actor.config.algorithm.target_networks.critic
    assert reference.config.algorithm.name == "three_network_ddpg"
    assert "target_networks" not in reference.config.algorithm
    assert reference.config.algorithm.agreement_mode == "none"
    assert reference.config.algorithm.update_schedule == "sequential"
    assert reference.config.algorithm.actor_state_source == "next"
    assert reference.config.algorithm.critic_initialization == "independent"
    assert reference.config.algorithm.beta_regularizer.mode == "none"
    assert reference.config.algorithm.beta_regularizer.coefficient == 0.0
    assert reference.config.algorithm.guard.validation_batch is False


def test_stage_c_varies_only_agreement_mode_and_documented_coefficients() -> None:
    planned = expand_manifest(_manifest("c"))
    expected_coefficients = {
        "three-network-none": ("none", 0.0, 0.0),
        "three-network-current": ("current", 0.0, 0.1),
        "three-network-next": ("next", 0.1, 0.0),
        "three-network-both": ("both", 0.1, 0.1),
    }
    fixed_paths = (
        "environment.id",
        "environment.wrappers",
        "environment.observation_normalization",
        "environment.reward_scale",
        "environment.action_bounds",
        "experiment.total_steps",
        "experiment.warmup_steps",
        "experiment.batch_size",
        "experiment.replay_capacity",
        "experiment.evaluation_frequency",
        "experiment.evaluation_episodes",
        "experiment.bootstrap_on_truncation",
        "experiment.clip_actions",
        "algorithm.hidden_layers",
        "algorithm.gamma",
        "algorithm.actor_learning_rate",
        "algorithm.critic_one_learning_rate",
        "algorithm.critic_two_learning_rate",
        "algorithm.exploration_noise.mode",
        "algorithm.exploration_noise.standard_deviation",
        "algorithm.exploration_noise.clip",
        "algorithm.action_rescaling",
        "algorithm.gradient_clip_norm",
        "algorithm.beta_regularizer.mode",
        "algorithm.beta_regularizer.coefficient",
        "algorithm.update_schedule",
        "algorithm.actor_state_source",
        "algorithm.critic_initialization",
        "algorithm.actor_update_delay",
        "algorithm.guard.validation_batch",
        "algorithm.device",
    )
    snapshots: list[dict[str, Any]] = []
    for variant_id, (mode, alpha, alpha_prime) in expected_coefficients.items():
        run = _seed_run(planned, variant_id)
        assert run.config.algorithm.name == "three_network_ddpg"
        assert "target_networks" not in run.config.algorithm
        assert run.config.algorithm.agreement_mode == mode
        assert run.config.algorithm.alpha == alpha
        assert run.config.algorithm.alpha_prime == alpha_prime
        assert run.config.algorithm.beta_regularizer.mode == "none"
        assert run.config.algorithm.beta_regularizer.mode != "l2_y_z"
        assert run.config.algorithm.guard.validation_batch is False
        snapshots.append({path: OmegaConf.select(run.config, path) for path in fixed_paths})
    assert snapshots == [snapshots[0]] * len(snapshots)


def test_filled_pilot_plan_records_workload_gates_and_interpretation_limits() -> None:
    plan = OmegaConf.to_container(OmegaConf.load(PLAN_PATH), resolve=True)
    assert isinstance(plan, dict)
    assert plan["training_seeds"] == list(DEVELOPMENT_TRAINING_SEEDS)
    assert plan["evaluation_seeds"] == list(EVALUATION_SEEDS)
    assert not set(plan["training_seeds"]) & set(plan["held_out_confirmatory_training_seeds"])
    stages = {stage["id"]: stage for stage in plan["stages"]}
    assert stages["stage-a"]["workload"]["planned_runs"] == 3
    assert stages["stage-b"]["workload"]["planned_runs"] == 9
    assert stages["stage-c"]["workload"]["planned_runs"] == 12
    assert stages["stage-b"]["gate_to_next_stage"].startswith("Inspect all Stage B")
    forbidden = plan["preregistration"]["forbidden_interpretations"]
    assert any("Lower shared TD error" in item for item in forbidden)
    assert any("monitoring-only" in item for item in forbidden)
    assert any("l2_y_z" in item for item in forbidden)
