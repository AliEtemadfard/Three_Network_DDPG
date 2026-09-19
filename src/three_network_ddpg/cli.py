"""Inspection-only CLI; training commands are intentionally absent."""

import argparse
from collections.abc import Sequence

from omegaconf import OmegaConf

from .algorithms.registry import ALGORITHMS
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="three-network-ddpg",
        description="Research scaffold: inspect algorithms and validate configuration; no training yet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("algorithms", help="List algorithm capability metadata and status")
    config_parser = sub.add_parser("config", help="Compose and inspect packaged YAML groups")
    config_parser.add_argument("action", choices=("show", "validate"))
    config_parser.add_argument("--algorithm", choices=tuple(ALGORITHMS))
    config_parser.add_argument("--environment", help="Environment config group name")
    config_parser.add_argument("--experiment", help="Experiment config group name")
    config_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "algorithms":
        for name, capabilities in ALGORITHMS.items():
            mode = "on-policy" if capabilities.is_on_policy else "off-policy"
            print(f"{name}\t{capabilities.implementation_status}\t{mode}")
        return 0
    try:
        config = load_config(
            algorithm=args.algorithm,
            environment=args.environment,
            experiment=args.experiment,
            overrides=args.overrides,
        )
    except ValueError as exc:
        build_parser().error(str(exc))
    if args.action == "show":
        print(OmegaConf.to_yaml(config, resolve=True), end="")
    else:
        print("Configuration valid (algorithm implementation remains planned).")
    return 0
