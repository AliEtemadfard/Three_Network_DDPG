"""Run setup records a real Git commit and the resolved run seed."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from three_network_ddpg.config import load_config
from three_network_ddpg.core.metadata import git_commit, prepare_run_directory


def test_run_directory_records_commit_and_resolved_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(environment="tiny_continuous")
    output = prepare_run_directory(tmp_path, config, seed=31, repo_root=repo_root)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    resolved = OmegaConf.load(output / "resolved_config.yaml")
    assert metadata["git_commit"] == git_commit(repo_root)
    assert metadata["git_commit"] is not None
    assert metadata["environment_id"] == "ThreeNetworkTinyContinuous-v0"
    assert metadata["seed"] == resolved.algorithm.seed == 31
    assert metadata["config"]["algorithm"]["seed"] == 31
    assert "gymnasium" in metadata["package_versions"]
    assert "cuda_devices" in metadata["hardware"]
