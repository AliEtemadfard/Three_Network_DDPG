"""Inspection and random-environment smoke CLI; no learning command exists."""

import argparse
from collections.abc import Sequence

from omegaconf import OmegaConf

from .algorithms.registry import ALGORITHMS
from .config import load_config
from .core.seeds import seed_environment, seed_everything
from .environments import make_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="three-network-ddpg",
        description="Research core: inspect configs and exercise environments; no learning algorithm yet.",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "algorithms":
        for name, capabilities in ALGORITHMS.items():
            mode = "on-policy" if capabilities.is_on_policy else "off-policy"
            print(f"{name}\t{capabilities.implementation_status}\t{mode}")
        return 0
    try:
        if args.command == "env-smoke":
            _environment_smoke(args.environment, args.steps, args.seed)
            return 0
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
