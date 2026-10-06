"""Failure-aware statistics and environment-step curve alignment."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .storage import atomic_write_json, read_status

SUPPORTED_METRICS = {
    "evaluation_episode_return",
    "evaluation_return_mean",
    "evaluation_episode_length",
    "evaluation_episode_length_mean",
    "episode_return",
    "episode_length",
    "shared_squared_error",
    "signed_residual_mean",
    "absolute_residual_mean",
    "q2_signed_residual_mean",
    "q2_absolute_residual_mean",
    "q2_td_squared_error",
    "current_agreement_error",
    "next_agreement_error",
    "q_mean",
    "q_absolute_mean",
    "q_abs_mean",
    "q_abs_max",
    "y_mean",
    "y1_mean",
    "z1_mean",
    "z_mean",
    "actor_gradient_norm",
    "critic_gradient_norm",
    "critic_one_gradient_norm",
    "critic_two_gradient_norm",
    "guard_accepted",
    "guard_rejected",
    "guard_retry_count",
    "guard_effective_learning_rate",
    "guard_validation_delta",
    "parameter_count",
    "wall_time_seconds",
    "updates_per_second",
    "environment_steps_per_second",
    "peak_memory_bytes",
}


def summary_statistics(
    values: list[float],
    *,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "iqr": None,
            "confidence_interval": None,
        }
    q25, q75 = np.percentile(finite, [25, 75])
    result: dict[str, Any] = {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "iqr": float(q75 - q25),
        "confidence_interval": None,
    }
    if finite.size >= 2:
        if bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        generator = np.random.default_rng(bootstrap_seed)
        indices = generator.integers(0, finite.size, size=(bootstrap_samples, finite.size))
        bootstrap_means = np.mean(finite[indices], axis=1)
        alpha = (1.0 - confidence_level) / 2.0
        low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
        result["confidence_interval"] = [float(low), float(high)]
    return result


def align_curves(
    curves: list[list[tuple[int, float]]],
) -> tuple[list[int], list[list[float | None]]]:
    """Align on the union of environment steps using linear interpolation only in-range."""
    steps = sorted({step for curve in curves for step, _ in curve})
    aligned: list[list[float | None]] = []
    for curve in curves:
        ordered = sorted(curve)
        if not ordered:
            aligned.append([None] * len(steps))
            continue
        x = np.asarray([point[0] for point in ordered], dtype=np.float64)
        y = np.asarray([point[1] for point in ordered], dtype=np.float64)
        row: list[float | None] = []
        for step in steps:
            if step < x[0] or step > x[-1]:
                row.append(None)
            else:
                row.append(float(np.interp(step, x, y)))
        aligned.append(row)
    return steps, aligned


def normalized_auc(curve: list[tuple[int, float]]) -> float | None:
    finite = [(step, value) for step, value in sorted(curve) if math.isfinite(value)]
    if not finite:
        return None
    if len(finite) == 1:
        return float(finite[0][1])
    x = np.asarray([point[0] for point in finite], dtype=np.float64)
    y = np.asarray([point[1] for point in finite], dtype=np.float64)
    width = float(x[-1] - x[0])
    if width <= 0:
        return float(np.mean(y))
    return float(np.trapezoid(y, x=x) / width)


def _read_metrics(path: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    non_finite = False
    if not path.exists():
        return records, non_finite
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = record.get("value")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            records.append(record)
        else:
            non_finite = True
    return records, non_finite


def aggregate_study(
    study_root: Path,
    *,
    last_k: int = 5,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate completed finite runs while reporting every planned run state."""
    if last_k < 1:
        raise ValueError("last_k must be positive")
    matrix = json.loads((study_root / "matrix.json").read_text(encoding="utf-8"))
    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in matrix:
        variants[str(spec["variant_id"])].append(spec)

    report: dict[str, Any] = {
        "study_root": str(study_root),
        "alignment_policy": (
            "union of environment steps; linear interpolation within each run range; no extrapolation"
        ),
        "failed_runs_are_omitted": False,
        "failed_runs_included": True,
        "variants": {},
    }
    for variant_id, specs in sorted(variants.items()):
        counts = {
            "planned_runs": len(specs),
            "started_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "divergent_runs": 0,
            "non_finite_runs": 0,
            "interrupted_runs": 0,
            "incomplete_runs": 0,
        }
        run_rows: list[dict[str, Any]] = []
        curves_by_metric: dict[str, list[tuple[str, list[tuple[int, float]]]]] = defaultdict(list)
        final_returns: list[float] = []
        auc_values: list[float] = []
        for spec in specs:
            run_root = study_root / "runs" / spec["run_id"]
            status_record = read_status(run_root)
            status = str(status_record.get("status", "incomplete"))
            attempt = int(status_record.get("attempt", 0))
            if attempt > 0:
                counts["started_runs"] += 1
            count_key = {
                "completed": "completed_runs",
                "failed": "failed_runs",
                "divergent": "divergent_runs",
                "non_finite": "non_finite_runs",
                "interrupted": "interrupted_runs",
                "incomplete": "incomplete_runs",
                "running": "incomplete_runs",
            }[status]
            counts[count_key] += 1
            row = {
                "run_id": spec["run_id"],
                "training_seed": spec["training_seed"],
                "status": status,
                "failure_reason": status_record.get("failure_reason"),
                "attempt": attempt,
                "non_finite_metrics": False,
            }
            if attempt > 0:
                metrics, non_finite = _read_metrics(
                    run_root / "attempts" / f"{attempt:04d}" / "metrics.jsonl"
                )
                row["non_finite_metrics"] = non_finite
                if status == "completed" and not non_finite:
                    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
                    for record in metrics:
                        name = str(record.get("name"))
                        if name in SUPPORTED_METRICS:
                            grouped[name].append((int(record["step"]), float(record["value"])))
                    for name, curve in grouped.items():
                        curves_by_metric[name].append((str(spec["run_id"]), sorted(curve)))
                    evaluation = grouped.get("evaluation_return_mean", [])
                    if evaluation:
                        final_returns.append(float(np.mean([value for _, value in evaluation[-last_k:]])))
                        auc = normalized_auc(evaluation)
                        if auc is not None:
                            auc_values.append(auc)
            run_rows.append(row)

        metric_reports: dict[str, Any] = {}
        for name, named_curves in sorted(curves_by_metric.items()):
            steps, aligned = align_curves([curve for _, curve in named_curves])
            aggregate_points: list[dict[str, Any]] = []
            for index, step in enumerate(steps):
                values = [row[index] for row in aligned if row[index] is not None]
                aggregate_points.append(
                    {"step": step}
                    | summary_statistics(
                        [float(value) for value in values],
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        bootstrap_seed=bootstrap_seed + step,
                    )
                )
            metric_reports[name] = {
                "individual_runs": [
                    {
                        "run_id": run_id,
                        "points": [{"step": step, "value": value} for step, value in curve],
                    }
                    for run_id, curve in named_curves
                ],
                "aggregate": aggregate_points,
            }
        report["variants"][variant_id] = {
            "counts": counts,
            "completion_rate": counts["completed_runs"] / counts["planned_runs"],
            "failure_counts": {
                key: counts[key]
                for key in (
                    "failed_runs",
                    "divergent_runs",
                    "non_finite_runs",
                    "interrupted_runs",
                    "incomplete_runs",
                )
            },
            "runs": run_rows,
            "final_return_last_k": summary_statistics(
                final_returns,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
            ),
            "normalized_auc": summary_statistics(
                auc_values,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed + 1,
            ),
            "metrics": metric_reports,
        }
    reports = study_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "aggregate.json", report)
    return report
