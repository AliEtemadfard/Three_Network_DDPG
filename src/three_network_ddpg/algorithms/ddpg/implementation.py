"""Faithful four-network Deep Deterministic Policy Gradient baseline."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ...core.actions import clip_action, scale_action_tensor
from ...core.batches import Transition, TransitionBatch
from ...core.interfaces import (
    ActionSelection,
    AlgorithmCapabilities,
    MetricRecord,
    OffPolicyAlgorithm,
    UpdateResult,
)


@dataclass(frozen=True, slots=True)
class DDPGSettings:
    observation_dim: int
    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    hidden_layers: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    gamma: float = 0.99
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 0.0
    tau: float = 0.005
    exploration_noise_std: float = 0.1
    exploration_noise_clip: float | None = None
    gradient_clip_norm: float | None = None
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if len(self.action_low) != self.action_dim or len(self.action_high) != self.action_dim:
            raise ValueError("action bounds must match action_dim")
        if any(low >= high for low, high in zip(self.action_low, self.action_high, strict=True)):
            raise ValueError("each action lower bound must be below its upper bound")
        if not self.hidden_layers or any(width <= 0 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if self.activation not in {"relu", "tanh", "elu"}:
            raise ValueError("unsupported activation")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if self.actor_learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.optimizer_epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer epsilon must be positive and weight decay nonnegative")
        if self.exploration_noise_std < 0:
            raise ValueError("exploration noise standard deviation must be nonnegative")
        if self.exploration_noise_clip is not None and self.exploration_noise_clip <= 0:
            raise ValueError("exploration_noise_clip must be positive when configured")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when configured")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")


def _activation(name: str) -> type[nn.Module]:
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[name]


def build_mlp(
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    *,
    activation: str,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Construct the configurable MLP used by both actor and critic."""
    layers: list[nn.Module] = []
    previous = input_dim
    activation_type = _activation(activation)
    for width in hidden_layers:
        layers.extend((nn.Linear(previous, width), activation_type()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """Tanh policy whose public output is already in environment coordinates."""

    def __init__(self, settings: DDPGSettings) -> None:
        super().__init__()
        self.network = build_mlp(
            settings.observation_dim,
            settings.hidden_layers,
            settings.action_dim,
            activation=settings.activation,
            output_activation=nn.Tanh(),
        )
        self.register_buffer("action_low", torch.tensor(settings.action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(settings.action_high, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return scale_action_tensor(self.network(observations), self.action_low, self.action_high)


class QCritic(nn.Module):
    """State-action critic returning one scalar per transition."""

    def __init__(self, settings: DDPGSettings) -> None:
        super().__init__()
        self.network = build_mlp(
            settings.observation_dim + settings.action_dim,
            settings.hidden_layers,
            1,
            activation=settings.activation,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observations, actions), dim=-1))


def ddpg_td_target(
    rewards: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute the DDPG target from the replay buffer's explicit mask."""
    if rewards.shape != bootstrap_mask.shape or rewards.shape != next_q.shape:
        raise ValueError("reward, bootstrap mask, and next Q shapes must match")
    return rewards + gamma * bootstrap_mask * next_q


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Polyak update target <- (1-tau) target + tau online."""
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True):
            target_parameter.lerp_(online_parameter, tau)


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [parameter.grad.detach().pow(2).sum() for parameter in parameters if parameter.grad is not None]
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


class DDPG(OffPolicyAlgorithm):
    """Standard DDPG with online/target actor and critic networks."""

    capabilities = AlgorithmCapabilities("ddpg", False, True, True, "verified")

    def __init__(self, settings: DDPGSettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        torch.manual_seed(settings.seed)
        self.actor = DeterministicActor(settings).to(self.device)
        self.critic = QCritic(settings).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=settings.actor_learning_rate,
            eps=settings.optimizer_epsilon,
            weight_decay=settings.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=settings.critic_learning_rate,
            eps=settings.optimizer_epsilon,
            weight_decay=settings.weight_decay,
        )
        self._noise_rng = np.random.default_rng(settings.seed)
        self._evaluation_mode = False
        self._update_step = 0
        self._environment_steps = 0
        self._action_low = np.asarray(settings.action_low, dtype=np.float32)
        self._action_high = np.asarray(settings.action_high, dtype=np.float32)

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "DDPG":
        hidden = tuple(int(width) for width in algorithm_config["hidden_layers"])
        noise = algorithm_config["exploration_noise"]
        target = algorithm_config["target_networks"]
        optimizer = algorithm_config["optimizer"]
        gradient_clip = algorithm_config.get("gradient_clip_norm")
        settings = DDPGSettings(
            observation_dim=observation_dim,
            action_dim=int(action_low.size),
            action_low=tuple(float(value) for value in action_low),
            action_high=tuple(float(value) for value in action_high),
            hidden_layers=hidden,
            activation=str(algorithm_config["activation"]),
            gamma=float(algorithm_config["gamma"]),
            actor_learning_rate=float(algorithm_config["actor_learning_rate"]),
            critic_learning_rate=float(algorithm_config["critic_one_learning_rate"]),
            optimizer_epsilon=float(optimizer["epsilon"]),
            weight_decay=float(optimizer["weight_decay"]),
            tau=float(target["tau"]),
            exploration_noise_std=float(noise["standard_deviation"]),
            exploration_noise_clip=(None if noise["clip"] is None else float(noise["clip"])),
            gradient_clip_norm=None if gradient_clip is None else float(gradient_clip),
            device=str(algorithm_config["device"]),
            seed=int(algorithm_config["seed"]),
        )
        return cls(settings)

    def select_action(self, observation: np.ndarray, *, deterministic: bool) -> ActionSelection:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.settings.observation_dim,):
            raise ValueError("observation has the wrong shape")
        with torch.no_grad():
            tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
            action = self.actor(tensor).squeeze(0).cpu().numpy()
        exploration = (
            not deterministic and not self._evaluation_mode and self.settings.exploration_noise_std > 0
        )
        if exploration:
            noise = self._noise_rng.normal(
                0.0, self.settings.exploration_noise_std, size=action.shape
            ).astype(np.float32)
            if self.settings.exploration_noise_clip is not None:
                noise = np.clip(
                    noise,
                    -self.settings.exploration_noise_clip,
                    self.settings.exploration_noise_clip,
                )
            action = action + noise
        action = clip_action(action, self._action_low, self._action_high)
        return ActionSelection(action, exploration_applied=exploration)

    def observe_transition(self, transition: Transition) -> None:
        self._environment_steps += 1

    def update_from_batch(
        self,
        batch: TransitionBatch,
        *,
        validation_batch: TransitionBatch | None = None,
    ) -> UpdateResult:
        if validation_batch is not None:
            raise ValueError("DDPG does not consume a validation batch during updates")
        batch = batch.to(self.device)
        with torch.no_grad():
            next_actions = self.target_actor(batch.next_observations)
            next_q = self.target_critic(batch.next_observations, next_actions)
            targets = ddpg_td_target(batch.rewards, batch.bootstrap_mask, next_q, self.settings.gamma)

        self.critic_optimizer.zero_grad(set_to_none=True)
        predicted_q = self.critic(batch.observations, batch.actions)
        critic_loss = torch.nn.functional.mse_loss(predicted_q, targets)
        critic_loss.backward()
        critic_parameters = tuple(self.critic.parameters())
        critic_gradient_norm = _gradient_norm(critic_parameters)
        if self.settings.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(critic_parameters, self.settings.gradient_clip_norm)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_actions = self.actor(batch.observations)
        actor_q = self.critic(batch.observations, actor_actions)
        actor_loss = -actor_q.mean()
        actor_loss.backward()
        actor_parameters = tuple(self.actor.parameters())
        actor_gradient_norm = _gradient_norm(actor_parameters)
        if self.settings.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(actor_parameters, self.settings.gradient_clip_norm)
        self.actor_optimizer.step()
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        soft_update(self.target_actor, self.actor, self.settings.tau)
        soft_update(self.target_critic, self.critic, self.settings.tau)
        self._update_step += 1
        step = self._update_step
        values = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "q_mean": float(predicted_q.detach().mean().cpu()),
            "q_abs_max": float(predicted_q.detach().abs().max().cpu()),
            "target_q_mean": float(targets.detach().mean().cpu()),
            "critic_gradient_norm": critic_gradient_norm,
            "actor_gradient_norm": actor_gradient_norm,
            "environment_steps": float(self._environment_steps),
            "update_steps": float(step),
        }
        return UpdateResult(
            metrics=tuple(MetricRecord(name, value, step, "train") for name, value in values.items()),
            updates_applied=1,
        )

    def set_evaluation_mode(self, enabled: bool) -> None:
        self._evaluation_mode = enabled
        self.actor.train(not enabled)
        self.critic.train(not enabled)
        self.target_actor.eval()
        self.target_critic.eval()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "settings": asdict(self.settings),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self._update_step,
            "environment_steps": self._environment_steps,
            "noise_rng_state": self._noise_rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._update_step = int(state["update_step"])
        self._environment_steps = int(state["environment_steps"])
        self._noise_rng.bit_generator.state = state["noise_rng_state"]

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(self.state_dict()), path)

    def restore_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state)
