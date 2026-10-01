"""Run selection, safe resume, and bounded worker orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..core.metadata import build_run_metadata
from ..experiment import run_off_policy_experiment_to_directory
from .manifest import ExperimentManifest, PlannedRun, expand_manifest
from .storage import (
    NonFiniteMetricError,
    atomic_write_json,
    begin_attempt,
    finish_attempt,
    initialize_study,
    metric_appender,
    metric_file_has_non_finite,
    read_status,
    utc_now,
)

RunExecutor = Callable[[PlannedRun, Path, Path], None]


def eligible_runs(
    planned: tuple[PlannedRun, ...],
    study_root: Path,
    *,
    resume: bool,
) -> tuple[PlannedRun, ...]:
    """Select only never-started runs, or retryable states for resume."""
    allowed = {"incomplete", "failed", "interrupted"} if resume else {"incomplete"}
    return tuple(
        run for run in planned if str(read_status(study_root / "runs" / run.run_id).get("status")) in allowed
    )


def _metadata(run: PlannedRun, repo_root: Path) -> dict[str, Any]:
    value = build_run_metadata(run.config, seed=run.training_seed, repo_root=repo_root)
    value.update(
        {
            "variant_id": run.variant_id,
            "variant_hash": run.variant_hash,
            "run_id": run.run_id,
            "configuration_hash": run.config_hash,
            "training_seed": run.training_seed,
            "evaluation_seeds": list(run.evaluation_seeds),
            "completion_status": "running",
            "failure_reason": None,
        }
    )
    return value


def execute_run(run: PlannedRun, study_root: Path, repo_root: Path) -> None:
    """Execute one run as a new attempt and publish one atomic terminal marker."""
    run_root = study_root / "runs" / run.run_id
    attempt, attempt_root = begin_attempt(run_root)
    metadata = _metadata(run, repo_root)
    atomic_write_json(attempt_root / "metadata.json", metadata)
    metrics_path = attempt_root / "metrics.jsonl"
    try:
        run_off_policy_experiment_to_directory(
            run.config,
            repo_root=repo_root,
            run_directory=attempt_root,
            evaluation_seeds=run.evaluation_seeds,
            metric_sink=metric_appender(metrics_path),
        )
        if metric_file_has_non_finite(metrics_path):
            metadata.update(
                {
                    "completion_status": "non_finite",
                    "failure_reason": "non-finite metric recorded",
                    "finished_at_utc": utc_now(),
                }
            )
            atomic_write_json(attempt_root / "metadata.json", metadata)
            finish_attempt(
                run_root,
                attempt_root,
                "non_finite",
                failure_reason="non-finite metric recorded",
            )
        else:
            metadata.update({"completion_status": "completed", "finished_at_utc": utc_now()})
            atomic_write_json(attempt_root / "metadata.json", metadata)
            finish_attempt(run_root, attempt_root, "completed")
    except KeyboardInterrupt:
        metadata.update(
            {
                "completion_status": "interrupted",
                "failure_reason": "interrupted by operator",
                "finished_at_utc": utc_now(),
            }
        )
        atomic_write_json(attempt_root / "metadata.json", metadata)
        finish_attempt(
            run_root,
            attempt_root,
            "interrupted",
            failure_reason="interrupted by operator",
        )
        raise
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        lower_reason = reason.lower()
        if (
            isinstance(exc, NonFiniteMetricError)
            or "non-finite" in lower_reason
            or "finite value" in lower_reason
        ):
            status = "non_finite"
        elif isinstance(exc, FloatingPointError | OverflowError):
            status = "divergent"
        else:
            status = "failed"
        metadata.update({"completion_status": status, "failure_reason": reason, "finished_at_utc": utc_now()})
        atomic_write_json(attempt_root / "metadata.json", metadata)
        finish_attempt(run_root, attempt_root, status, failure_reason=reason)
        raise


def _worker(payload: dict[str, Any]) -> tuple[str, str | None]:
    run = PlannedRun(
        variant_id=payload["variant_id"],
        variant_hash=payload["variant_hash"],
        run_id=payload["run_id"],
        training_seed=payload["training_seed"],
        evaluation_seeds=tuple(payload["evaluation_seeds"]),
        config=OmegaConf.create(payload["config"]),
        config_hash=payload["config_hash"],
    )
    try:
        execute_run(run, Path(payload["study_root"]), Path(payload["repo_root"]))
    except Exception as exc:
        return run.run_id, f"{type(exc).__name__}: {exc}"
    return run.run_id, None


def execute_manifest(
    manifest: ExperimentManifest,
    *,
    repo_root: Path,
    output_root: Path | None = None,
    workers: int | None = None,
    resume: bool = False,
    restart: bool = False,
    executor: RunExecutor | None = None,
) -> tuple[Path, dict[str, str | None]]:
    """Freeze, select, and execute a manifest without hiding failed runs."""
    planned = expand_manifest(manifest)
    study_root = initialize_study(
        manifest,
        planned,
        repo_root=repo_root,
        output_root=output_root,
        restart=restart,
    )
    selected = eligible_runs(planned, study_root, resume=resume)
    worker_count = workers if workers is not None else manifest.workers
    if worker_count < 1:
        raise ValueError("workers must be positive")
    outcomes: dict[str, str | None] = {}
    if executor is not None:
        for run in selected:
            try:
                executor(run, study_root, repo_root)
            except Exception as exc:
                outcomes[run.run_id] = f"{type(exc).__name__}: {exc}"
            else:
                outcomes[run.run_id] = None
        return study_root, outcomes
    if worker_count == 1:
        for run in selected:
            try:
                execute_run(run, study_root, repo_root)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                outcomes[run.run_id] = f"{type(exc).__name__}: {exc}"
            else:
                outcomes[run.run_id] = None
        return study_root, outcomes

    payloads = [
        run.as_dict()
        | {
            "variant_hash": run.variant_hash,
            "study_root": str(study_root.resolve()),
            "repo_root": str(repo_root.resolve()),
        }
        for run in selected
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_worker, payload): payload["run_id"] for payload in payloads}
        try:
            for future in as_completed(futures):
                run_id, error = future.result()
                outcomes[run_id] = error
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    return study_root, outcomes


def matrix_rows(manifest: ExperimentManifest) -> list[dict[str, Any]]:
    return [run.as_dict() for run in expand_manifest(manifest)]


def load_frozen_matrix(study_root: Path) -> list[dict[str, Any]]:
    value = json.loads((study_root / "matrix.json").read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("frozen matrix is invalid")
    return value
