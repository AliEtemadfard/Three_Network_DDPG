"""Atomic status markers, compatibility checks, and only-missing resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from three_network_ddpg.protocol.manifest import expand_manifest, load_manifest
from three_network_ddpg.protocol.orchestration import eligible_runs
from three_network_ddpg.protocol.storage import (
    NonFiniteMetricError,
    append_jsonl,
    atomic_write_json,
    begin_attempt,
    finish_attempt,
    initialize_study,
    metric_appender,
    read_status,
)


def _manifest(tmp_path: Path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        f"""schema_version: 2
name: storage-test
description: synthetic
defaults:
  environment: tiny_continuous
  experiment: ddpg_baseline
training_seeds: [1, 2]
evaluation_seeds: [101]
variants:
  - id: ddpg
    algorithm: ddpg
    overrides:
      - experiment.total_steps=4
      - experiment.warmup_steps=1
      - experiment.batch_size=2
      - experiment.evaluation_frequency=2
      - experiment.evaluation_episodes=1
execution:
  output_root: {str(tmp_path / "outputs").replace(chr(92), "/")}
  workers: 1
analysis: {{}}
""",
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    return manifest, expand_manifest(manifest)


def test_atomic_json_and_append_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "value.json"
    atomic_write_json(target, {"complete": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}
    stream = tmp_path / "metrics.jsonl"
    append_jsonl(stream, {"value": 1})
    append_jsonl(stream, {"value": 2})
    assert [json.loads(line)["value"] for line in stream.read_text().splitlines()] == [1, 2]
    assert not list(tmp_path.glob("*.tmp"))


def test_statuses_and_atomic_completion_markers(tmp_path: Path) -> None:
    manifest, planned = _manifest(tmp_path)
    root = initialize_study(manifest, planned, repo_root=tmp_path)
    run_root = root / "runs" / planned[0].run_id
    attempt, attempt_root = begin_attempt(run_root)
    assert attempt == 1
    assert read_status(run_root)["status"] == "running"
    finish_attempt(run_root, attempt_root, "failed", failure_reason="synthetic")
    status = read_status(run_root)
    assert status["status"] == "failed"
    assert status["failure_reason"] == "synthetic"
    assert json.loads((attempt_root / "FAILED.json").read_text())["status"] == "failed"

    completed_root = root / "runs" / planned[1].run_id
    _, completed_attempt = begin_attempt(completed_root)
    finish_attempt(completed_root, completed_attempt, "completed")
    assert read_status(completed_root)["status"] == "completed"
    assert (completed_attempt / "COMPLETED.json").exists()


def test_only_missing_and_resume_selection(tmp_path: Path) -> None:
    manifest, planned = _manifest(tmp_path)
    root = initialize_study(manifest, planned, repo_root=tmp_path)
    first_root = root / "runs" / planned[0].run_id
    _, attempt_root = begin_attempt(first_root)
    finish_attempt(first_root, attempt_root, "failed", failure_reason="synthetic")
    assert [run.run_id for run in eligible_runs(planned, root, resume=False)] == [planned[1].run_id]
    assert {run.run_id for run in eligible_runs(planned, root, resume=True)} == {
        planned[0].run_id,
        planned[1].run_id,
    }


def test_incompatible_resume_is_rejected_and_restart_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, planned = _manifest(tmp_path)
    monkeypatch.setattr("three_network_ddpg.protocol.storage.git_commit", lambda _: "first")
    root = initialize_study(manifest, planned, repo_root=tmp_path)
    monkeypatch.setattr("three_network_ddpg.protocol.storage.git_commit", lambda _: "second")
    with pytest.raises(ValueError, match="incompatible existing study"):
        initialize_study(manifest, planned, repo_root=tmp_path)
    replacement = initialize_study(manifest, planned, repo_root=tmp_path, restart=True)
    assert replacement == root
    assert list(root.parent.glob("storage-test.restart-*"))


def test_dead_running_process_is_reported_interrupted(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    atomic_write_json(
        run_root / "status.json",
        {
            "status": "running",
            "attempt": 1,
            "pid": 2147483647,
            "hostname": __import__("socket").gethostname(),
        },
    )
    assert read_status(run_root)["status"] == "interrupted"


def test_live_running_process_is_not_signalled(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    atomic_write_json(
        run_root / "status.json",
        {
            "status": "running",
            "attempt": 1,
            "pid": os.getpid(),
            "hostname": __import__("socket").gethostname(),
        },
    )
    assert read_status(run_root)["status"] == "running"


def test_non_finite_metric_is_rejected_before_json_append(tmp_path: Path) -> None:
    append = metric_appender(tmp_path / "metrics.jsonl")
    with pytest.raises(NonFiniteMetricError, match="non-finite metric"):
        append(SimpleNamespace(name="loss", value=float("nan"), step=3, phase="train"))
    assert not (tmp_path / "metrics.jsonl").exists()
