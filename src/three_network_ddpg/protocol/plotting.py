"""Dependency-light SVG learning-curve plots with visible seed traces."""

# ruff: noqa: E501

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from .storage import atomic_write_text

_WIDTH = 1120
_HEIGHT = 800
_LEFT = 155.0
_RIGHT = 1060.0
_TOP = 110.0
_BOTTOM = 560.0
_COMPARISON_METRICS = frozenset({"evaluation_return_mean"})
_METRIC_LABELS = {"evaluation_return_mean": "Evaluation return"}
_VARIANT_LABELS = {
    "ddpg-pipeline-smoke": "DDPG",
    "ddpg-no-target-actor-pipeline-smoke": "DDPG without target actor",
    "three-network-shared-error-pipeline-smoke": "Three-network shared-error sequential",
}


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _variant_label(variant_id: str) -> str:
    return _VARIANT_LABELS.get(variant_id, variant_id.replace("-", " ").title())


def _ticks(lower: float, upper: float) -> list[float]:
    """Return compact, readable numerical ticks for an axis range."""
    span = max(upper - lower, 1e-12)
    raw = span / 5
    power = 10.0 ** math.floor(math.log10(raw))
    fraction = raw / power
    step = (1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0) * power
    first = math.floor(lower / step) * step
    last = math.ceil(upper / step) * step
    values: list[float] = []
    value = first
    while value <= last + step * 1e-9:
        values.append(value)
        value += step
    return values


def _tick_label(value: float) -> str:
    if abs(value) >= 1000 or 0 < abs(value) < 0.01:
        return f"{value:.2g}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3g}"


def _expanded_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x_padding = max((x1 - x0) * 0.04, 1.0)
    y_padding = max((y1 - y0) * 0.08, max(abs(y0), abs(y1), 1.0) * 0.02)
    return x0 - x_padding, x1 + x_padding, y0 - y_padding, y1 + y_padding


def _coordinate_x(value: float, bounds: tuple[float, float, float, float]) -> float:
    x0, x1, _, _ = bounds
    return _LEFT + (value - x0) / max(x1 - x0, 1e-12) * (_RIGHT - _LEFT)


def _coordinate_y(value: float, bounds: tuple[float, float, float, float]) -> float:
    _, _, y0, y1 = bounds
    return _TOP + (y1 - value) / max(y1 - y0, 1e-12) * (_BOTTOM - _TOP)


def _points(values: list[tuple[float, float]], bounds: tuple[float, float, float, float]) -> str:
    return " ".join(f"{_coordinate_x(x, bounds):.2f},{_coordinate_y(y, bounds):.2f}" for x, y in values)


def _axes(bounds: tuple[float, float, float, float], y_label: str) -> str:
    x0, x1, y0, y1 = bounds
    elements: list[str] = []
    for tick in _ticks(x0, x1):
        coordinate = _coordinate_x(tick, bounds)
        elements.extend(
            (
                f'<line x1="{coordinate:.2f}" y1="{_TOP}" x2="{coordinate:.2f}" y2="{_BOTTOM}" stroke="#d8dde5"/>',
                f'<text x="{coordinate:.2f}" y="{_BOTTOM + 25:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{_tick_label(tick)}</text>',
            )
        )
    for tick in _ticks(y0, y1):
        coordinate = _coordinate_y(tick, bounds)
        elements.extend(
            (
                f'<line x1="{_LEFT}" y1="{coordinate:.2f}" x2="{_RIGHT}" y2="{coordinate:.2f}" stroke="#d8dde5"/>',
                f'<text x="{_LEFT - 12:.2f}" y="{coordinate + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{_tick_label(tick)}</text>',
            )
        )
    elements.extend(
        (
            f'<line x1="{_LEFT}" y1="{_BOTTOM}" x2="{_RIGHT}" y2="{_BOTTOM}" stroke="#333"/>',
            f'<line x1="{_LEFT}" y1="{_TOP}" x2="{_LEFT}" y2="{_BOTTOM}" stroke="#333"/>',
            f'<text x="{(_LEFT + _RIGHT) / 2:.2f}" y="{_BOTTOM + 65:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14">Environment steps</text>',
            f'<text x="30" y="{(_TOP + _BOTTOM) / 2:.2f}" text-anchor="middle" transform="rotate(-90 30 {(_TOP + _BOTTOM) / 2:.2f})" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        )
    )
    return "\n".join(elements)


def _state_text(counts: dict[str, Any]) -> str:
    return (
        f"completed={counts['completed_runs']}, failed={counts['failed_runs']}, "
        f"divergent={counts['divergent_runs']}, non_finite={counts['non_finite_runs']}, "
        f"interrupted={counts['interrupted_runs']}, incomplete={counts['incomplete_runs']}"
    )


def _confidence_band(
    aggregate: list[dict[str, Any]], bounds: tuple[float, float, float, float], color: str
) -> str:
    values = [
        point
        for point in aggregate
        if point.get("confidence_interval") is not None and point.get("mean") is not None
    ]
    if len(values) < 2:
        return ""
    upper = [(float(point["step"]), float(point["confidence_interval"][1])) for point in values]
    lower = [(float(point["step"]), float(point["confidence_interval"][0])) for point in reversed(values)]
    return f'<polygon class="confidence-band" points="{_points(upper + lower, bounds)}" fill="{color}" fill-opacity="0.16" stroke="none"/>'


def _series(variant_id: str, variant: dict[str, Any], metric: str, color: str) -> dict[str, Any] | None:
    metric_data = variant["metrics"].get(metric)
    if not metric_data:
        return None
    seeds = [
        [(float(point["step"]), float(point["value"])) for point in run["points"]]
        for run in metric_data["individual_runs"]
    ]
    mean = [
        (float(point["step"]), float(point["mean"]))
        for point in metric_data["aggregate"]
        if point["mean"] is not None
    ]
    if not mean or not any(seeds):
        return None
    return {
        "aggregate": metric_data["aggregate"],
        "color": color,
        "counts": variant["counts"],
        "label": _variant_label(variant_id),
        "mean": mean,
        "seeds": seeds,
    }


def _markers(values: list[tuple[float, float]], bounds: tuple[float, float, float, float], color: str) -> str:
    return "".join(
        f'<circle cx="{_coordinate_x(x, bounds):.2f}" cy="{_coordinate_y(y, bounds):.2f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>'
        for x, y in values
    )


def _render_svg(metric: str, series: list[dict[str, Any]], title: str, comparison: bool) -> str:
    all_points = [point for item in series for seed in item["seeds"] for point in seed]
    all_points.extend(point for item in series for point in item["mean"])
    bounds = _expanded_bounds(all_points)
    lines: list[str] = []
    annotations: list[str] = []
    for item in series:
        seed_count = len(item["seeds"])
        ci_available = any(point.get("confidence_interval") is not None for point in item["aggregate"])
        ci_text = "95% CI" if ci_available else "CI unavailable (requires n>=2)"
        annotations.append(f"{item['label']}: n={seed_count}; {ci_text}; {_state_text(item['counts'])}")
        if seed_count > 1:
            lines.extend(
                f'<polyline class="seed-curve" fill="none" stroke="{item["color"]}" stroke-opacity="0.22" stroke-width="1.5" points="{_points(seed, bounds)}"/>'
                for seed in item["seeds"]
            )
        band = _confidence_band(item["aggregate"], bounds, item["color"])
        if band:
            lines.append(band)
        lines.extend(
            (
                f'<polyline class="aggregate-curve" fill="none" stroke="{item["color"]}" stroke-width="3" points="{_points(item["mean"], bounds)}"/>',
                _markers(item["mean"], bounds, item["color"]),
            )
        )
    legend = ""
    if comparison:
        legend = "\n".join(
            f'<line x1="{_LEFT}" y1="{655 + index * 22}" x2="{_LEFT + 28}" y2="{655 + index * 22}" stroke="{item["color"]}" stroke-width="3"/><text x="{_LEFT + 36}" y="{659 + index * 22}" font-family="sans-serif" font-size="13">{html.escape(item["label"])}</text>'
            for index, item in enumerate(series)
        )
    annotation_rows = "\n".join(
        f'<text x="{_LEFT}" y="{730 + index * 18}" font-family="sans-serif" font-size="11">{html.escape(annotation)}</text>'
        for index, annotation in enumerate(annotations)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}">
<rect width="{_WIDTH}" height="{_HEIGHT}" fill="white"/>
<text x="{_LEFT}" y="35" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>
<text x="{_LEFT}" y="62" font-family="sans-serif" font-size="13">Points mark recorded observations; curves use absolute metric values.</text>
{_axes(bounds, _metric_label(metric))}
{chr(10).join(lines)}
{legend}
{annotation_rows}
</svg>
'''


def plot_report(
    report: dict[str, Any],
    output_directory: Path,
    *,
    metric: str = "evaluation_return_mean",
) -> list[Path]:
    """Write readable per-variant SVGs and compatible shared-scale comparisons."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = ("#3569b7", "#be5b3b", "#34805c", "#7a56a5")
    comparable: list[dict[str, Any]] = []
    for color_index, (variant_id, variant) in enumerate(sorted(report["variants"].items())):
        series = _series(variant_id, variant, metric, colors[color_index % len(colors)])
        if series is None:
            continue
        path = output_directory / f"{variant_id}-{metric}.svg"
        atomic_write_text(
            path,
            _render_svg(
                metric,
                [series],
                f"{series['label']}: {_metric_label(metric)}",
                comparison=False,
            ),
        )
        paths.append(path)
        comparable.append(series)
    if metric in _COMPARISON_METRICS and len(comparable) >= 2:
        comparison_path = output_directory / f"comparison-{metric}.svg"
        atomic_write_text(
            comparison_path,
            _render_svg(
                metric,
                comparable,
                f"{_metric_label(metric)} comparison",
                comparison=True,
            ),
        )
        paths.append(comparison_path)
    atomic_write_text(
        output_directory / "index.json",
        json.dumps(
            {
                "comparison_metrics": sorted(_COMPARISON_METRICS),
                "metric": metric,
                "plots": [path.name for path in paths],
            },
            indent=2,
        )
        + "\n",
    )
    return paths
