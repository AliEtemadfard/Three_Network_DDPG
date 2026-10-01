"""CLI for configurations, algorithms, and reproducible experiment protocols."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from .algorithms.registry import ALGORITHMS
from .config import load_config
from .core.seeds import seed_environment, seed_everything
from .environments import make_environment
from .experiment import (
    run_ddpg_experiment,
    run_ddpg_no_target_actor_experiment,
    run_three_network_experiment,
)
from .protocol.aggregation import aggregate_study
from .protocol.manifest import expand_manifest, load_manifest
from .protocol.orchestration import execute_manifest
from .protocol.plotting import plot_report
from .protocol.storage import status_counts, study_directory


def _add_train_arguments(parser: argparse.ArgumentParser, *, default_experiment: str) -> None:
    parser.add_argument(
        "--environment",
        choices=("tiny_continuous", "pendulum_v1"),
        default="pendulum_v1",
    )
    parser.add_argument("--experiment", default=default_experiment)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def _add_protocol_location(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="three-network-ddpg",
        description=(
            "Research CLI with verified DDPG, experimental variants, and a reproducible experiment protocol."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("algorithms", help="List algorithm capability metadata and status")
    config_parser = sub.add_parser("config", help="Compose and inspect packaged YAML groups")
    config_parser.add_argument("action", choices=("show", "validate"))
    config_parser.add_argument("--algorithm", choices=tuple(ALGORITHMS))
    config_parser.add_argument("--environment", help="Environment config group name")
    config_parser.add_argument("--experiment", help="Experiment config group name")
    config_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    smoke_parser = sub.add_parser("env-smoke", help="Step a seeded random policy; no learning")
    smoke_parser.add_argument("--environment", choices=("tiny_continuous", "pendulum_v1"), required=True)
    smoke_parser.add_argument("--steps", type=int, required=True)
    smoke_parser.add_argument("--seed", type=int, required=True)
    ddpg_parser = sub.add_parser("train-ddpg", help="Run one local DDPG experiment and save metadata")
    _add_train_arguments(ddpg_parser, default_experiment="ddpg_baseline")
    no_target_actor_parser = sub.add_parser(
        "train-ddpg-no-target-actor",
        help="Run the DDPG target-actor ablation and save metadata",
    )
    _add_train_arguments(no_target_actor_parser, default_experiment="ddpg_no_target_actor")
    proposed_parser = sub.add_parser(
        "train-three-network",
        help="Run the experimental three-network method and save metadata",
    )
    _add_train_arguments(proposed_parser, default_experiment="three_network_core")

    protocol_parser = sub.add_parser("experiment", help="Validate, execute, and analyze a manifest")
    protocol_sub = protocol_parser.add_subparsers(dest="experiment_action", required=True)
    for action, help_text in (
        ("validate", "Validate the manifest and every composed configuration"),
        ("plan", "Display the expanded run plan without training"),
        ("matrix", "Display the expanded matrix without training"),
        ("count", "Display the number of planned runs"),
        ("status", "Show run status counts"),
        ("aggregate", "Aggregate completed artifacts without hiding failures"),
        ("plot", "Generate SVG plots from aggregate artifacts"),
    ):
        action_parser = protocol_sub.add_parser(action, help=help_text)
        _add_protocol_location(action_parser)
    for action, help_text in (
        ("run", "Execute only never-started runs"),
        ("resume", "Execute incomplete, failed, or interrupted runs"),
    ):
        action_parser = protocol_sub.add_parser(action, help=help_text)
        _add_protocol_location(action_parser)
        action_parser.add_argument("--workers", type=int)
        action_parser.add_argument(
            "--restart",
            action="store_true",
            help="Archive an incompatible study directory before starting a new one",
        )
    plot_parser = protocol_sub.choices["plot"]
    plot_parser.add_argument("--metric", default="evaluation_return_mean")
    return parser


def _environment_smoke(environment_group: str, steps: int, seed: int) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive")
    config = load_config(environment=environment_group)
    environment = make_environment(config.environment)
    seed_everything(seed)
    completed = 0
    terminated_count = 0
    truncated_count = 0
    try:
        seed_environment(environment, seed)
        for index in range(steps):
            action = environment.action_space.sample()
            _, _, terminated, truncated, _ = environment.step(action)
            terminated_count += int(terminated)
            truncated_count += int(truncated)
            if terminated or truncated:
                completed += 1
                if index + 1 < steps:
                    seed_environment(environment, seed + completed)
    finally:
        environment.close()
    print(
        f"Random-policy smoke: steps={steps}, completed_episodes={completed}, "
        f"terminated={terminated_count}, truncated={truncated_count}"
    )


def _protocol_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    planned = expand_manifest(manifest)
    if args.experiment_action == "validate":
        print(
            f"Manifest valid: name={manifest.name}, variants={len(manifest.variants)}, "
            f"planned_runs={len(planned)}"
        )
        return 0
    if args.experiment_action in {"plan", "matrix"}:
        print(json.dumps([run.as_dict() for run in planned], indent=2, sort_keys=True))
        return 0
    if args.experiment_action == "count":
        print(len(planned))
        return 0
    root = study_directory(manifest, args.output_root)
    if args.experiment_action == "status":
        print(json.dumps(status_counts(root), indent=2, sort_keys=True))
        return 0
    if args.experiment_action in {"run", "resume"}:
        root, outcomes = execute_manifest(
            manifest,
            repo_root=Path.cwd(),
            output_root=args.output_root,
            workers=args.workers,
            resume=args.experiment_action == "resume",
            restart=args.restart,
        )
        failures = {run_id: error for run_id, error in outcomes.items() if error is not None}
        print(f"Study: {root}")
        print(json.dumps(status_counts(root), indent=2, sort_keys=True))
        if failures:
            print(json.dumps({"execution_errors": failures}, indent=2, sort_keys=True))
            return 1
        return 0
    analysis = manifest.analysis
    report_path = root / "reports" / "aggregate.json"
    if args.experiment_action == "aggregate":
        report = aggregate_study(
            root,
            last_k=int(analysis.get("last_k", 5)),
            bootstrap_samples=int(analysis.get("bootstrap_samples", 2000)),
            confidence_level=float(analysis.get("confidence_level", 0.95)),
            bootstrap_seed=int(analysis.get("bootstrap_seed", 0)),
        )
        print(f"Aggregate report: {root / 'reports' / 'aggregate.json'}")
        print(json.dumps({key: value["counts"] for key, value in report["variants"].items()}, indent=2))
        return 0
    if not report_path.exists():
        raise ValueError("aggregate report is missing; run experiment aggregate first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paths = plot_report(report, root / "reports" / "plots", metric=args.metric)
    print(f"Generated {len(paths)} plot(s) in {root / 'reports' / 'plots'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "algorithms":
        for name, capabilities in ALGORITHMS.items():
            mode = "on-policy" if capabilities.is_on_policy else "off-policy"
            print(f"{name}\t{capabilities.implementation_status}\t{mode}")
        return 0
    try:
        if args.command == "experiment":
            return _protocol_command(args)
        if args.command == "env-smoke":
            _environment_smoke(args.environment, args.steps, args.seed)
            return 0
        if args.command == "train-ddpg":
            config = load_config(
                algorithm="ddpg",
                environment=args.environment,
                experiment=args.experiment,
                overrides=args.overrides,
            )
            run_directory = run_ddpg_experiment(config, repo_root=Path.cwd())
            print(f"DDPG smoke artifacts: {run_directory}")
            print("This single run verifies execution and is not scientific evidence.")
            return 0
        if args.command == "train-ddpg-no-target-actor":
            config = load_config(
                algorithm="ddpg_no_target_actor",
                environment=args.environment,
                experiment=args.experiment,
                overrides=args.overrides,
            )
            run_directory = run_ddpg_no_target_actor_experiment(config, repo_root=Path.cwd())
            print(f"DDPG no-target-actor smoke artifacts: {run_directory}")
            print("This single run verifies execution and is not scientific evidence.")
            return 0
        if args.command == "train-three-network":
            config = load_config(
                algorithm="three_network_ddpg",
                environment=args.environment,
                experiment=args.experiment,
                overrides=args.overrides,
            )
            run_directory = run_three_network_experiment(config, repo_root=Path.cwd())
            print(f"Three-network smoke artifacts: {run_directory}")
            print("This single run verifies execution and is not scientific evidence.")
            return 0
        config = load_config(
            algorithm=args.algorithm,
            environment=args.environment,
            experiment=args.experiment,
            overrides=args.overrides,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.action == "show":
        print(OmegaConf.to_yaml(config, resolve=True), end="")
    else:
        status = config.algorithm.implementation_status
        print(f"Configuration valid (algorithm status: {status}).")
    return 0
