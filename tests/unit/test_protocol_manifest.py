"""Manifest expansion, pairing, IDs, and scientific guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from three_network_ddpg.cli import build_parser
from three_network_ddpg.protocol.manifest import canonical_hash, expand_manifest, load_manifest


def _write_manifest(path: Path, *, duplicate: bool = False, extra_override: str = "") -> Path:
    second_id = "baseline" if duplicate else "proposed"
    text = f"""schema_version: 2
name: test-study
description: synthetic
defaults:
  environment: tiny_continuous
  experiment: ddpg_baseline
training_seeds: [3, 7]
evaluation_seeds: [101, 102]
variants:
  - id: baseline
    algorithm: ddpg
    overrides:
      - experiment.total_steps=8
      - experiment.warmup_steps=1
      - experiment.batch_size=2
      - experiment.evaluation_frequency=4
      - experiment.evaluation_episodes=2
  - id: {second_id}
    algorithm: three_network_ddpg
    experiment: three_network_core
    overrides:
      - experiment.total_steps=8
      - experiment.warmup_steps=1
      - experiment.batch_size=2
      - experiment.evaluation_frequency=4
      - experiment.evaluation_episodes=2
      {extra_override}
execution:
  output_root: runs/tests
  workers: 1
analysis:
  last_k: 2
"""
    path.write_text(text, encoding="utf-8")
    return path


def test_matrix_pairs_seeds_and_ids_are_stable_unique(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path / "manifest.yaml")
    first = expand_manifest(load_manifest(path))
    second = expand_manifest(load_manifest(path))
    assert len(first) == 4
    assert [run.run_id for run in first] == [run.run_id for run in second]
    assert len({run.run_id for run in first}) == len(first)
    by_variant = {}
    for run in first:
        by_variant.setdefault(run.variant_id, []).append(run.training_seed)
        assert run.evaluation_seeds == (101, 102)
        assert run.training_seed not in run.evaluation_seeds
    assert by_variant == {"baseline": [3, 7], "proposed": [3, 7]}


def test_config_hash_changes_with_resolved_configuration(tmp_path: Path) -> None:
    first = expand_manifest(load_manifest(_write_manifest(tmp_path / "first.yaml")))
    changed_path = _write_manifest(tmp_path / "changed.yaml")
    text = changed_path.read_text(encoding="utf-8").replace(
        "experiment.total_steps=8", "experiment.total_steps=9", 1
    )
    changed_path.write_text(text, encoding="utf-8")
    changed = expand_manifest(load_manifest(changed_path))
    assert first[0].config_hash != changed[0].config_hash
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_duplicate_variant_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate variant id"):
        load_manifest(_write_manifest(tmp_path / "duplicate.yaml", duplicate=True))


def test_l2_y_z_is_blocked_before_training(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "invalid.yaml",
        extra_override="- algorithm.beta_regularizer.mode=l2_y_z",
    )
    with pytest.raises(ValueError, match="not an available scientific variant"):
        load_manifest(path)


def test_training_and_evaluation_seeds_must_be_distinct(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path / "overlap.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "evaluation_seeds: [101, 102]", "evaluation_seeds: [7, 102]"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distinct"):
        load_manifest(path)


def test_invalid_analysis_settings_are_rejected_during_validation(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path / "invalid-analysis.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace("last_k: 2", "last_k: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="analysis.last_k"):
        load_manifest(path)


def test_protocol_cli_exposes_complete_lifecycle() -> None:
    parser = build_parser()
    for action in ("validate", "plan", "run", "status", "resume", "aggregate", "plot"):
        args = parser.parse_args(["experiment", action, "manifest.yaml"])
        assert args.experiment_action == action


@pytest.mark.parametrize(
    ("name", "expected_runs"),
    (("smoke.yaml", 1), ("development.yaml", 21), ("confirmatory.yaml", 70)),
)
def test_shipped_manifests_expand(name: str, expected_runs: int) -> None:
    repository = Path(__file__).resolve().parents[2]
    manifest = load_manifest(repository / "experiments" / "manifests" / name)
    assert len(expand_manifest(manifest)) == expected_runs
