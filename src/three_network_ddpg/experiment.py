"""Executable off-policy experiment orchestration and local JSONL tracking."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .algorithms.registry import create_algorithm
from .core.devices import resolve_device
from .core.interfaces import MetricRecord, OffPolicyAlgorithm
from .core.metadata import prepare_run_directory
from .core.replay import ReplayBuffer
from .environments import make_environment
from .evaluation import EvaluationResult, evaluate
from .training import OffPolicyTrainer, TrainerSettings


def _run_off_policy_experiment(
    config: DictConfig,
    *,
    repo_root: Path,
    expected_algorithm: str,
) -> Path:
    if config.algorithm.name != expected_algorithm:
        raise ValueError(f"{expected_algorithm} experiment requires algorithm.name={expected_algorithm}")
    training_environment = make_environment(config.environment)
    evaluation_environment = make_environment(config.environment)
    try:
        if training_environment.observation_space.shape is None:
            raise ValueError("observation space must expose a shape")
        resolved_device = resolve_device(str(config.algorithm.device))
        runtime_algorithm_config = OmegaConf.merge(config.algorithm, {"device": resolved_device})
        algorithm = create_algorithm(
            expected_algorithm,
            algorithm_config=runtime_algorithm_config,
            observation_dim=int(training_environment.observation_space.shape[0]),
            action_low=training_environment.action_space.low,
            action_high=training_environment.action_space.high,
        )
        if not isinstance(algorithm, OffPolicyAlgorithm):
            raise TypeError("experiment runner requires an off-policy algorithm")
        replay = ReplayBuffer(
            capacity=int(config.experiment.replay_capacity),
            observation_shape=training_environment.observation_space.shape,
            action_shape=training_environment.action_space.shape,
            seed=int(config.algorithm.seed),
            bootstrap_on_truncation=bool(config.experiment.bootstrap_on_truncation),
            device=resolved_device,
        )
        evaluation_results: list[EvaluationResult] = []

        def evaluation_callback(
            environment_step: int, evaluated_algorithm: OffPolicyAlgorithm
        ) -> tuple[MetricRecord, ...]:
            result = evaluate(
                evaluated_algorithm,
                evaluation_environment,
                episodes=int(config.experiment.evaluation_episodes),
                seed=int(config.algorithm.seed) + 10000 + environment_step,
                clip_actions=bool(config.experiment.clip_actions),
            )
            evaluation_results.append(result)
            return tuple(
                MetricRecord(record.name, record.value, environment_step, record.phase)
                for record in result.metrics
            )

        settings = TrainerSettings(
            total_steps=int(config.experiment.total_steps),
            warmup_steps=int(config.experiment.warmup_steps),
            batch_size=int(config.experiment.batch_size),
            clip_actions=bool(config.experiment.clip_actions),
            bootstrap_on_truncation=bool(config.experiment.bootstrap_on_truncation),
            evaluation_frequency=int(config.experiment.evaluation_frequency),
        )
        run_directory = prepare_run_directory(
            Path(config.experiment.output_root),
            config,
            seed=int(config.algorithm.seed),
            repo_root=repo_root,
        )
        training = OffPolicyTrainer(
            algorithm=algorithm,
            environment=training_environment,
            replay=replay,
            settings=settings,
            evaluation_callback=evaluation_callback,
        ).run(seed=int(config.algorithm.seed), deterministic_algorithms=True)
        if not evaluation_results or (
            config.experiment.total_steps % config.experiment.evaluation_frequency != 0
        ):
            final_records = evaluation_callback(int(config.experiment.total_steps), algorithm)
        else:
            final_records = ()
        metrics = (*training.metrics, *final_records)
        with (run_directory / "metrics.jsonl").open("x", encoding="utf-8") as handle:
            for metric in metrics:
                handle.write(json.dumps(asdict(metric), sort_keys=True) + "\n")
        final_evaluation = evaluation_results[-1]
        summary: dict[str, Any] = {
            "scientific_evidence": False,
            "note": "Single smoke run for execution verification only.",
            "algorithm": expected_algorithm,
            "environment_steps": training.environment_steps,
            "update_steps": training.gradient_updates,
            "wall_time_seconds": training.wall_time_seconds,
            "training_episode_returns": training.completed_episode_returns,
            "training_episode_lengths": training.completed_episode_lengths,
            "evaluation_returns": final_evaluation.returns,
            "evaluation_lengths": final_evaluation.lengths,
            "evaluation_count": len(evaluation_results),
        }
        (run_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        algorithm.save_checkpoint(run_directory / "checkpoint.pt")
        return run_directory
    finally:
        training_environment.close()
        evaluation_environment.close()


def run_ddpg_experiment(config: DictConfig, *, repo_root: Path) -> Path:
    """Train and evaluate the verified DDPG baseline once."""
    return _run_off_policy_experiment(config, repo_root=repo_root, expected_algorithm="ddpg")


def run_ddpg_no_target_actor_experiment(config: DictConfig, *, repo_root: Path) -> Path:
    """Train and evaluate the DDPG target-actor ablation once."""
    return _run_off_policy_experiment(
        config,
        repo_root=repo_root,
        expected_algorithm="ddpg_no_target_actor",
    )


def run_three_network_experiment(config: DictConfig, *, repo_root: Path) -> Path:
    """Train and evaluate the experimental three-network method once."""
    return _run_off_policy_experiment(config, repo_root=repo_root, expected_algorithm="three_network_ddpg")
