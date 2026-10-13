"""Fast checks retained from the bootstrap scaffold."""

import importlib.metadata
import json
import platform
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
from three_network_ddpg.core.devices import resolve_device
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
    assert config.algorithm.name == name
    assert config.algorithm.implementation_status == capabilities.implementation_status
    assert config.algorithm.is_on_policy == capabilities.is_on_policy
    if name == "ddpg":
        assert capabilities.implementation_status == "verified"
    elif name in {
        "ddpg_no_target_actor",
        "three_network_ddpg",
        "three_network_shared_d",
        "three_network_shared_d_board",
        "three_network_shared_d_paired",
        "three_network_shared_d_conditioned_paired",
        "three_network_shared_d_conditioned",
        "three_network_shared_d_synchronous",
        "three_network_shared_d_two_timescale",
    }:
        assert capabilities.implementation_status == "experimental"
    else:
        assert capabilities.implementation_status == "planned"
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
    config = load_config(
        overrides=[
            "algorithm.agreement_mode=both",
            "algorithm.alpha=0.2",
            "algorithm.alpha_prime=0.3",
            "algorithm.beta_regularizer.mode=l2_y",
            "algorithm.beta_regularizer.coefficient=0.01",
            "algorithm.update_schedule=guarded",
            "algorithm.guard.scope=critics",
            "algorithm.actor_state_source=current",
            "algorithm.diagnostic_frequency=null",
        ]
    )
    assert config.algorithm.agreement_mode == "both"
    assert config.algorithm.beta_regularizer.mode == "l2_y"
    assert config.algorithm.update_schedule == "guarded"
    assert config.algorithm.guard.scope == "critics"
    assert config.algorithm.actor_state_source == "current"
    assert config.algorithm.diagnostic_frequency is None
    with pytest.raises(ValueError, match="current agreement does not use alpha"):
        load_config(
            overrides=[
                "algorithm.agreement_mode=current",
                "algorithm.alpha=0.1",
                "algorithm.alpha_prime=0.1",
            ]
        )
    with pytest.raises(ValueError, match="beta mode none"):
        load_config(overrides=["algorithm.beta_regularizer.coefficient=0.1"])
    with pytest.raises(ValueError, match="guard.scope"):
        load_config(overrides=["algorithm.guard.scope=invalid"])


def test_cli_help_and_inspection(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "algorithms" in capsys.readouterr().out
    assert main(["algorithms"]) == 0
    output = capsys.readouterr().out
    assert "a3c\tplanned\ton-policy" in output
    assert "three_network_ddpg\texperimental\toff-policy" in output
    assert "ddpg_no_target_actor\texperimental\toff-policy" in output
    assert main(["config", "validate"]) == 0
    assert "Configuration valid" in capsys.readouterr().out


def test_seed_utility_repeats_python_sequence() -> None:
    report = seed_everything(17)
    first = random.random()
    seed_everything(17)
    assert random.random() == first
    assert report.seed == 17
    assert report.numpy_seeded
    assert report.torch_seeded
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


def test_run_metadata_contains_resolved_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(overrides=["algorithm.device=auto"])
    directory = prepare_run_directory(tmp_path, config, seed=5, repo_root=tmp_path)
    assert (directory / "metadata.json").exists()
    assert (directory / "resolved_config.yaml").exists()
    resolved = OmegaConf.load(directory / "resolved_config.yaml")
    metadata = json.loads((directory / "metadata.json").read_text())
    assert resolved.algorithm.name == "three_network_ddpg"
    assert resolved.algorithm.seed == 5
    assert metadata["requested_device"] == "auto"
    assert metadata["resolved_device"] == "cpu"
    assert metadata["python_version"] == platform.python_version()
    assert metadata["python_implementation"] == platform.python_implementation()
    assert metadata["platform"] == platform.platform()
    for distribution in ("torch", "gymnasium", "numpy"):
        assert metadata["package_versions"][distribution] == importlib.metadata.version(distribution)


def test_auto_device_resolves_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_auto_device_resolves_to_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
