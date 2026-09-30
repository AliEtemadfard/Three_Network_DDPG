"""Reproducible run-directory naming and provenance capture."""

import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _slug(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe run path component: {value!r}")
    return value


def git_commit(repo_root: Path) -> str | None:
    """Return HEAD without changing Git configuration; an unborn branch yields None."""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root.resolve()}", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _hardware_metadata() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_devices": [],
    }
    try:
        import torch
    except ImportError:
        return hardware
    if torch.cuda.is_available():
        hardware["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return hardware


def build_run_metadata(
    config: DictConfig,
    *,
    seed: int,
    repo_root: Path,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """Collect fields required beside every future run."""
    started = started_at or datetime.now(UTC)
    resolved = OmegaConf.merge(config, {"algorithm": {"seed": seed}})
    packages: dict[str, str | None] = {}
    for name in ("three-network-ddpg", "torch", "gymnasium", "omegaconf", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "started_at_utc": started.astimezone(UTC).isoformat(),
        "seed": seed,
        "git_commit": git_commit(repo_root),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "algorithm": resolved.algorithm.name,
        "environment_id": resolved.environment.id,
        "environment_wrappers": OmegaConf.to_container(resolved.environment.wrappers, resolve=True),
        "config": OmegaConf.to_container(resolved, resolve=True),
        "package_versions": packages,
        "hardware": _hardware_metadata(),
    }


def prepare_run_directory(
    output_root: Path,
    config: DictConfig,
    *,
    seed: int,
    repo_root: Path,
    started_at: datetime | None = None,
) -> Path:
    """Create a new run directory with resolved config and provenance."""
    started = started_at or datetime.now(UTC)
    resolved = OmegaConf.merge(config, {"algorithm": {"seed": seed}})
    run_id = f"{started.astimezone(UTC):%Y%m%dT%H%M%SZ}-seed-{seed}"
    path = (
        output_root
        / _slug(resolved.experiment.name)
        / _slug(resolved.algorithm.name)
        / _slug(resolved.environment.id)
        / run_id
    )
    path.mkdir(parents=True, exist_ok=False)
    metadata = build_run_metadata(resolved, seed=seed, repo_root=repo_root, started_at=started)
    with (path / "resolved_config.yaml").open("x", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(resolved, resolve=True))
    with (path / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
