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


def _plot_variant(curves: list[list[tuple[int, float]]], *, metric: str = "evaluation_return_mean") -> dict:
    aggregate = []
    for index, step in enumerate(curves[0]):
        values = [curve[index][1] for curve in curves]
        aggregate.append(
            {
                "step": step[0],
                "mean": sum(values) / len(values),
                "confidence_interval": [min(values), max(values)] if len(curves) > 1 else None,
            }
        )
    return {
        "counts": {
            "completed_runs": len(curves),
            "divergent_runs": 1,
            "failed_runs": 1,
            "incomplete_runs": 0,
            "interrupted_runs": 0,
            "non_finite_runs": 1,
        },
        "metrics": {
            metric: {
                "aggregate": aggregate,
                "individual_runs": [
                    {"points": [{"step": step, "value": value} for step, value in curve]} for curve in curves
                ],
            }
        },
    }


def test_plots_are_readable_shared_scale_and_honest_about_seed_count(tmp_path: Path) -> None:
    curves = [[(500, -1700.0), (1000, -1300.0), (2000, -900.0)]]
    report = {
        "variants": {
            "ddpg-pipeline-smoke": _plot_variant(curves),
            "ddpg-no-target-actor-pipeline-smoke": _plot_variant(
                [[(500, -1800.0), (1000, -1500.0), (2000, -1200.0)]]
            ),
            "three-network-shared-error-pipeline-smoke": _plot_variant(
                [[(500, -1100.0), (1000, -1400.0), (2000, -1600.0)]]
            ),
        }
    }
    plots = tmp_path / "plots"
    paths = plot_report(report, plots)
    assert len(paths) == 4
    comparison = plots / "comparison-evaluation_return_mean.svg"
    svg = comparison.read_text(encoding="utf-8")
    assert "Environment steps" in svg and "Evaluation return" in svg
    assert ">-" in svg and 'viewBox="0 0 1120 800"' in svg
    assert svg.count('font-size="12"') >= 4 and svg.count("<circle") == 9
    assert 'y="766"' in svg
    assert "DDPG without target actor" in svg
    assert "Three-network shared-error sequential" in svg
    assert "n=1; CI unavailable (requires n&gt;=2)" in svg
    assert "completed=1, failed=1, divergent=1, non_finite=1" in svg
    assert "confidence-band" not in svg
    plot_report(report, plots)
    assert svg == comparison.read_text(encoding="utf-8")

    mechanism_report = {
        "variants": {
            variant_id: _plot_variant(curves, metric="shared_squared_error")
            for variant_id in report["variants"]
        }
    }
    mechanism_paths = plot_report(mechanism_report, tmp_path / "mechanism", metric="shared_squared_error")
    assert len(mechanism_paths) == 3
    assert not (tmp_path / "mechanism" / "comparison-shared_squared_error.svg").exists()


def test_multi_seed_plot_has_faint_seed_curves_and_confidence_band(tmp_path: Path) -> None:
    report = {
        "variants": {
            "v": _plot_variant(
                [
                    [(500, -1600.0), (1000, -1200.0), (2000, -1000.0)],
                    [(500, -1700.0), (1000, -1300.0), (2000, -1100.0)],
                ]
            )
        }
    }
    path = plot_report(report, tmp_path / "plots")[0]
    svg = path.read_text(encoding="utf-8")
    assert svg.count('class="seed-curve"') == 2
    assert 'stroke-opacity="0.22"' in svg
    assert 'class="confidence-band"' in svg
    assert "n=2; 95% CI" in svg
