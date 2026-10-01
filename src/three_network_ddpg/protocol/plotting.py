"""Dependency-light SVG learning-curve plots with visible seed traces."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .storage import atomic_write_text


def _points(values: list[tuple[float, float]], bounds: tuple[float, float, float, float]) -> str:
    x0, x1, y0, y1 = bounds
    width, height = 760.0, 400.0
    return " ".join(
        f"{70 + (x - x0) / max(x1 - x0, 1e-12) * width:.2f},"
        f"{30 + (y1 - y) / max(y1 - y0, 1e-12) * height:.2f}"
        for x, y in values
    )


def plot_report(
    report: dict[str, Any],
    output_directory: Path,
    *,
    metric: str = "evaluation_return_mean",
) -> list[Path]:
    """Write one SVG per variant; non-completed runs remain in the subtitle."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = ("#3569b7", "#be5b3b", "#34805c", "#7a56a5")
    for color_index, (variant_id, variant) in enumerate(sorted(report["variants"].items())):
        metric_data = variant["metrics"].get(metric)
        if not metric_data:
            continue
        individual = metric_data["individual_runs"]
        aggregate = metric_data["aggregate"]
        all_points = [
            (float(point["step"]), float(point["value"])) for run in individual for point in run["points"]
        ]
        mean_points = [
            (float(point["step"]), float(point["mean"])) for point in aggregate if point["mean"] is not None
        ]
        if not all_points or not mean_points:
            continue
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        bounds = (min(xs), max(xs), min(ys), max(ys))
        color = colors[color_index % len(colors)]
        seed_lines = "\n".join(
            f'<polyline fill="none" stroke="{color}" stroke-opacity="0.22" stroke-width="1.5" '
            f'points="{_points([(float(p["step"]), float(p["value"])) for p in run["points"]], bounds)}"/>'
            for run in individual
        )
        mean_line = (
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{_points(mean_points, bounds)}"/>'
        )
        counts = variant["counts"]
        subtitle = (
            f"completed={counts['completed_runs']}, failed={counts['failed_runs']}, "
            f"divergent={counts['divergent_runs']}, non_finite={counts['non_finite_runs']}, "
            f"interrupted={counts['interrupted_runs']}, "
            f"incomplete={counts['incomplete_runs']}"
        )
        title = html.escape(f"{variant_id}: {metric}")
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">
<rect width="900" height="500" fill="white"/>
<text x="70" y="20" font-family="sans-serif" font-size="16">{title}</text>
<text x="70" y="465" font-family="sans-serif" font-size="12">{html.escape(subtitle)}</text>
<line x1="70" y1="430" x2="830" y2="430" stroke="#333"/>
<line x1="70" y1="30" x2="70" y2="430" stroke="#333"/>
{seed_lines}
{mean_line}
</svg>
"""
        path = output_directory / f"{variant_id}-{metric}.svg"
        atomic_write_text(path, svg)
        paths.append(path)
    atomic_write_text(
        output_directory / "index.json",
        json.dumps({"metric": metric, "plots": [path.name for path in paths]}, indent=2) + "\n",
    )
    return paths
