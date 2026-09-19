"""Fast checks retained from the bootstrap scaffold."""

import random

import pytest
import torch
from omegaconf import OmegaConf

import three_network_ddpg
from three_network_ddpg.algorithms.registry import (
    ALGORITHMS,
    create_algorithm,
    get_algorithm_metadata,
)
from three_network_ddpg.cli import main
from three_network_ddpg.config import load_config, validate_config
from three_network_ddpg.core.batches import RolloutBatch, TransitionBatch
from three_network_ddpg.core.metadata import prepare_run_directory
from three_network_ddpg.core.seeds import seed_environment, seed_everything


def test_imports_and_typed_batches() -> None:
    assert three_network_ddpg.__version__ == "0.1.0"
    flags = torch.tensor([[False]], dtype=torch.bool)
    transition = TransitionBatch(
        observations=torch.zeros((1, 2)),
        actions=torch.zeros((1, 1)),
        rewards=torch.zeros((1, 1)),
        next_observations=torch.ones((1, 2)),
        terminated=flags,
        truncated=flags.clone(),
        bootstrap_mask=torch.ones((1, 1)),
    )
    rollout = RolloutBatch(
        observations=torch.zeros((1, 2)),
        actions=torch.zeros((1, 1)),
        rewards=torch.zeros((1, 1)),
        next_observations=torch.ones((1, 2)),
        terminated=flags,
        truncated=flags.clone(),
    )
    assert transition.batch_size == 1
    assert not rollout.terminated.any()


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_registry_metadata_and_composition(name: str) -> None:
    capabilities = get_algorithm_metadata(name)
    config = load_config(algorithm=name)
    assert capabilities.implementation_status == "planned"
    assert config.algorithm.name == name
    assert config.algorithm.implementation_status == capabilities.implementation_status
    assert config.algorithm.is_on_policy == capabilities.is_on_policy
    with pytest.raises(NotImplementedError, match="planned"):
        create_algorithm(name)


def test_proposed_method_has_no_target_networks() -> None:
    config = load_config()
    assert "target_networks" not in config.algorithm
    assert config.algorithm.actor_state_source == "next"
    config.algorithm.target_networks = {"actor": True, "critic": True}
    with pytest.raises(ValueError, match="must not define target"):
        validate_config(config)


def test_configuration_override_validation() -> None:
    config = load_config(overrides=["algorithm.agreement_mode=current", "algorithm.alpha_prime=0.5"])
    assert config.algorithm.agreement_mode == "current"
    assert config.algorithm.alpha_prime == 0.5
    with pytest.raises(ValueError, match="backtracking_factor"):
        load_config(overrides=["algorithm.guard.backtracking_factor=1"])


def test_cli_help_and_inspection(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "algorithms" in capsys.readouterr().out
    assert main(["algorithms"]) == 0
    assert "a3c\tplanned\ton-policy" in capsys.readouterr().out
    assert main(["config", "validate"]) == 0
    assert "Configuration valid" in capsys.readouterr().out


def test_seed_utility_repeats_python_sequence() -> None:
    report = seed_everything(17)
    first = random.random()
    seed_everything(17)
    assert random.random() == first
    assert report.seed == 17
    with pytest.raises(ValueError, match="seed must"):
        seed_everything(-1)


def test_environment_seeding() -> None:
    class Space:
        def __init__(self) -> None:
            self.received = None

        def seed(self, value: int) -> None:
            self.received = value

    class Environment:
        def __init__(self) -> None:
            self.action_space = Space()

        def reset(self, *, seed: int) -> tuple[int, dict[str, int]]:
            return seed, {"seed": seed}

    environment = Environment()
    assert seed_environment(environment, 3) == (3, {"seed": 3})
    assert environment.action_space.received == 3


def test_run_metadata_contains_resolved_config(tmp_path) -> None:
    config = load_config()
    directory = prepare_run_directory(tmp_path, config, seed=5, repo_root=tmp_path)
    assert (directory / "metadata.json").exists()
    assert (directory / "resolved_config.yaml").exists()
    resolved = OmegaConf.load(directory / "resolved_config.yaml")
    assert resolved.algorithm.name == "three_network_ddpg"
    assert resolved.algorithm.seed == 5
