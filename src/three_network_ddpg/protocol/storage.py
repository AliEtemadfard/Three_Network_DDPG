"""Crash-safe artifact storage and run-state reconciliation."""

from __future__ import annotations

import json
import math
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

from ..core.interfaces import MetricRecord
from ..core.metadata import git_commit
from .manifest import ExperimentManifest, PlannedRun

RunStatus = Literal[
    "incomplete",
    "running",
    "completed",
    "failed",
    "divergent",
    "non_finite",
    "interrupted",
]


class NonFiniteMetricError(FloatingPointError):
    """Raised before a non-finite metric can be serialized as invalid JSON."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file atomically using a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    """Append one fsynced JSON record with a single operating-system write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short JSONL append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def metric_appender(path: Path):
    def append(record: MetricRecord) -> None:
        if not math.isfinite(float(record.value)):
            raise NonFiniteMetricError(f"non-finite metric {record.name!r} at step {record.step}")
        append_jsonl(
            path,
            {"name": record.name, "value": record.value, "step": record.step, "phase": record.phase},
        )

    return append


def study_directory(manifest: ExperimentManifest, output_root: Path | None = None) -> Path:
    return (output_root or manifest.output_root) / manifest.name


def _archive_existing(path: Path) -> Path:
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = path.with_name(f"{path.name}.restart-{suffix}")
    os.replace(path, destination)
    return destination


def initialize_study(
    manifest: ExperimentManifest,
    planned: tuple[PlannedRun, ...],
    *,
    repo_root: Path,
    output_root: Path | None = None,
    restart: bool = False,
) -> Path:
    """Freeze a manifest and matrix, refusing incompatible reuse."""
    root = study_directory(manifest, output_root)
    commit = git_commit(repo_root)
    lock = {
        "schema_version": manifest.schema_version,
        "manifest_name": manifest.name,
        "manifest_hash": manifest.manifest_hash,
        "git_commit": commit,
        "created_at_utc": utc_now(),
    }
    lock_path = root / "manifest.lock.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        mismatches = [key for key in ("manifest_hash", "git_commit") if existing.get(key) != lock.get(key)]
        if mismatches and not restart:
            joined = ", ".join(mismatches)
            raise ValueError(
                f"incompatible existing study ({joined}); use --restart or a different --output-root"
            )
        if mismatches:
            _archive_existing(root)
    root.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        atomic_write_json(lock_path, lock)
        atomic_write_json(root / "manifest.source.json", manifest.source)
        atomic_write_json(root / "matrix.json", [run.as_dict() for run in planned])
    for run in planned:
        run_root = root / "runs" / run.run_id
        run_root.mkdir(parents=True, exist_ok=True)
        spec_path = run_root / "run_spec.json"
        if spec_path.exists():
            existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if (
                existing_spec.get("config_hash") != run.config_hash
                or existing_spec.get("git_commit") != commit
            ):
                raise ValueError(
                    f"run {run.run_id} is incompatible with its config or Git commit; "
                    "use --restart or a different --output-root"
                )
        else:
            spec = run.as_dict() | {"git_commit": commit}
            atomic_write_json(spec_path, spec)
            atomic_write_text(
                run_root / "resolved_config.yaml",
                OmegaConf.to_yaml(run.config, resolve=True),
            )
            write_status(run_root, "incomplete", failure_reason=None, attempt=0)
    return root


def write_status(
    run_root: Path,
    status: RunStatus,
    *,
    failure_reason: str | None,
    attempt: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "failure_reason": failure_reason,
        "attempt": attempt,
        "updated_at_utc": utc_now(),
    }
    if status == "running":
        payload.update({"pid": os.getpid(), "hostname": socket.gethostname()})
    if extra:
        payload.update(extra)
    atomic_write_json(run_root / "status.json", payload)


def read_status(run_root: Path) -> dict[str, Any]:
    path = run_root / "status.json"
    if not path.exists():
        return {"status": "incomplete", "failure_reason": None, "attempt": 0}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") == "running":
        same_host = value.get("hostname") == socket.gethostname()
        pid = value.get("pid")
        if same_host and isinstance(pid, int) and not _pid_alive(pid):
            value = value | {"status": "interrupted", "failure_reason": "runner process is no longer active"}
    return value


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def begin_attempt(run_root: Path) -> tuple[int, Path]:
    status = read_status(run_root)
    attempt = int(status.get("attempt", 0)) + 1
    attempt_root = run_root / "attempts" / f"{attempt:04d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    write_status(run_root, "running", failure_reason=None, attempt=attempt)
    atomic_write_json(
        attempt_root / "STARTED.json",
        {"status": "running", "attempt": attempt, "started_at_utc": utc_now()},
    )
    return attempt, attempt_root


def finish_attempt(
    run_root: Path,
    attempt_root: Path,
    status: Literal["completed", "failed", "divergent", "non_finite", "interrupted"],
    *,
    failure_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    attempt = int(attempt_root.name)
    payload: dict[str, Any] = {
        "status": status,
        "attempt": attempt,
        "failure_reason": failure_reason,
        "finished_at_utc": utc_now(),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(attempt_root / f"{status.upper()}.json", payload)
    write_status(run_root, status, failure_reason=failure_reason, attempt=attempt, extra=extra)


def metric_file_has_non_finite(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        metric = value.get("value")
        if isinstance(metric, int | float) and not math.isfinite(float(metric)):
            return True
    return False


def status_counts(study_root: Path) -> dict[str, int]:
    counts = {
        key: 0
        for key in (
            "incomplete",
            "running",
            "completed",
            "failed",
            "divergent",
            "non_finite",
            "interrupted",
        )
    }
    runs_root = study_root / "runs"
    if not runs_root.exists():
        return counts
    for run_root in runs_root.iterdir():
        if run_root.is_dir():
            status = str(read_status(run_root).get("status", "incomplete"))
            counts[status] = counts.get(status, 0) + 1
    return counts
