"""Synthetic aggregation, non-finite handling, interpolation, and plots."""

from __future__ import annotations

import json
from pathlib import Path

from three_network_ddpg.protocol.aggregation import (
    aggregate_study,
    align_curves,
    summary_statistics,
)
from three_network_ddpg.protocol.plotting import plot_report
from three_network_ddpg.protocol.storage import atomic_write_json


def test_confidence_interval_requires_two_valid_seeds() -> None:
    assert summary_statistics([1.0])["confidence_interval"] is None
    result = summary_statistics([1.0, 3.0], bootstrap_samples=200, bootstrap_seed=4)
    assert result["confidence_interval"] is not None
    assert result["mean"] == 2.0
    assert result["standard_deviation"] > 0


def test_interpolation_has_no_extrapolation() -> None:
    steps, aligned = align_curves([[(0, 0.0), (10, 10.0)], [(5, 10.0), (10, 20.0)]])
    assert steps == [0, 5, 10]
    assert aligned[0] == [0.0, 5.0, 10.0]
    assert aligned[1] == [None, 10.0, 20.0]


def _run(
    study: Path,
    *,
    run_id: str,
    seed: int,
    status: str,
    values: list[tuple[int, str]],
    failure_reason: str | None = None,
) -> dict:
    run_root = study / "runs" / run_id
    attempt_root = run_root / "attempts" / "0001"
    attempt_root.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "name": "evaluation_return_mean",
                "value": json.loads(value),
                "step": step,
                "phase": "evaluation",
            }
        )
        for step, value in values
    ]
    (attempt_root / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(
        run_root / "status.json",
        {"status": status, "attempt": 1, "failure_reason": failure_reason},
    )
    return {"variant_id": "v", "run_id": run_id, "training_seed": seed}


def test_aggregation_retains_failures_nonfinite_and_plots_seed_traces(tmp_path: Path) -> None:
    study = tmp_path / "study"
    first = _run(study, run_id="v-seed-1-a", seed=1, status="completed", values=[(10, "1.0"), (20, "3.0")])
    second = _run(study, run_id="v-seed-2-b", seed=2, status="completed", values=[(10, "2.0"), (20, "4.0")])
    failed = _run(
        study,
        run_id="v-seed-3-c",
        seed=3,
        status="failed",
        values=[],
        failure_reason="synthetic",
    )
    non_finite = _run(
        study,
        run_id="v-seed-4-d",
        seed=4,
        status="non_finite",
        values=[(10, "NaN")],
        failure_reason="non-finite metric",
    )
    divergent = _run(
        study,
        run_id="v-seed-5-e",
        seed=5,
        status="divergent",
        values=[],
        failure_reason="synthetic divergence",
    )
    atomic_write_json(study / "matrix.json", [first, second, failed, non_finite, divergent])
    report = aggregate_study(study, last_k=1, bootstrap_samples=200, bootstrap_seed=5)
    variant = report["variants"]["v"]
    assert variant["counts"] == {
        "planned_runs": 5,
        "started_runs": 5,
        "completed_runs": 2,
        "failed_runs": 1,
        "divergent_runs": 1,
        "non_finite_runs": 1,
        "interrupted_runs": 0,
        "incomplete_runs": 0,
    }
    assert variant["completion_rate"] == 0.4
    assert variant["failure_counts"]["non_finite_runs"] == 1
    assert variant["final_return_last_k"]["mean"] == 3.5
    assert variant["final_return_last_k"]["confidence_interval"] is not None
    non_finite_row = next(row for row in variant["runs"] if row["status"] == "non_finite")
    assert non_finite_row["non_finite_metrics"] is True
    paths = plot_report(report, study / "reports" / "plots")
    assert len(paths) == 1
    svg = paths[0].read_text(encoding="utf-8")
    assert 'stroke-opacity="0.22"' in svg
    assert "failed=1" in svg and "divergent=1" in svg and "non_finite=1" in svg
