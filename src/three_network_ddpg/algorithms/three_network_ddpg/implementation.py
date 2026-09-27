"""Minimal target-free three-network actor-critic with shared TD error only."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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

UpdateSchedule = Literal["sequential", "simultaneous"]
ActorStateSource = Literal["current", "next"]


@dataclass(frozen=True, slots=True)
class ThreeNetworkSettings:
    observation_dim: int
    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    hidden_layers: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    gamma: float = 0.99
    actor_learning_rate: float = 3e-4
    critic_one_learning_rate: float = 1e-3
    critic_two_learning_rate: float = 1e-3
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 0.0
    update_schedule: UpdateSchedule = "sequential"
    actor_state_source: ActorStateSource = "next"
    exploration_noise_std: float = 0.1
    exploration_noise_clip: float | None = None
    gradient_clip_norm: float | None = None
    diagnostic_frequency: int | None = 100
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
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        for name in (
            "actor_learning_rate",
            "critic_one_learning_rate",
            "critic_two_learning_rate",
            "optimizer_epsilon",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.update_schedule not in {"sequential", "simultaneous"}:
            raise ValueError("update_schedule must be sequential or simultaneous")
        if self.actor_state_source not in {"current", "next"}:
            raise ValueError("actor_state_source must be current or next")
        if self.exploration_noise_std < 0:
            raise ValueError("exploration_noise_std must be nonnegative")
        if self.exploration_noise_clip is not None and self.exploration_noise_clip <= 0:
            raise ValueError("exploration_noise_clip must be positive when configured")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when configured")
        if self.diagnostic_frequency is not None and self.diagnostic_frequency <= 0:
            raise ValueError("diagnostic_frequency must be positive when configured")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")

    @classmethod
    def from_config(
        cls,
        algorithm_config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "ThreeNetworkSettings":
        """Build the implemented subset while rejecting unsupported research options."""
        if algorithm_config["agreement_mode"] != "none":
            raise ValueError("minimal three-network core does not implement agreement penalties")
        if float(algorithm_config["alpha"]) != 0 or float(algorithm_config["alpha_prime"]) != 0:
            raise ValueError("minimal three-network core requires alpha and alpha_prime to be zero")
        beta = algorithm_config["beta_regularizer"]
        if beta["mode"] != "none" or float(beta["coefficient"]) != 0:
            raise ValueError("minimal three-network core does not implement beta regularization")
        if algorithm_config["critic_initialization"] != "independent":
            raise ValueError("minimal three-network core requires independent critic initialization")
        if int(algorithm_config["actor_update_delay"]) != 1:
            raise ValueError("minimal three-network core updates the actor on every learner step")
        if algorithm_config["optimizer"]["name"] != "adam":
            raise ValueError("minimal three-network core currently supports Adam")
        if not bool(algorithm_config["action_rescaling"]):
            raise ValueError("minimal three-network core requires action rescaling")
        noise = algorithm_config["exploration_noise"]
        if noise["mode"] not in {"gaussian", "none"}:
            raise ValueError("minimal three-network core supports gaussian or disabled exploration")
        gradient_clip = algorithm_config.get("gradient_clip_norm")
        diagnostic_frequency = algorithm_config.get("diagnostic_frequency")
        return cls(
            observation_dim=observation_dim,
            action_dim=int(action_low.size),
            action_low=tuple(float(value) for value in action_low),
            action_high=tuple(float(value) for value in action_high),
            hidden_layers=tuple(int(width) for width in algorithm_config["hidden_layers"]),
            activation=str(algorithm_config["activation"]),
            gamma=float(algorithm_config["gamma"]),
            actor_learning_rate=float(algorithm_config["actor_learning_rate"]),
            critic_one_learning_rate=float(algorithm_config["critic_one_learning_rate"]),
            critic_two_learning_rate=float(algorithm_config["critic_two_learning_rate"]),
            optimizer_epsilon=float(algorithm_config["optimizer"]["epsilon"]),
            weight_decay=float(algorithm_config["optimizer"]["weight_decay"]),
            update_schedule=str(algorithm_config["update_schedule"]),
            actor_state_source=str(algorithm_config["actor_state_source"]),
            exploration_noise_std=(0.0 if noise["mode"] == "none" else float(noise["standard_deviation"])),
            exploration_noise_clip=None if noise["clip"] is None else float(noise["clip"]),
            gradient_clip_norm=None if gradient_clip is None else float(gradient_clip),
            diagnostic_frequency=(None if diagnostic_frequency is None else int(diagnostic_frequency)),
            device=str(algorithm_config["device"]),
            seed=int(algorithm_config["seed"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an exact, constructor-compatible configuration snapshot."""
        return asdict(self)


def _activation(name: str) -> type[nn.Module]:
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[name]


def _build_mlp(
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    *,
    activation: str,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
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


class ThreeNetworkActor(nn.Module):
    """Bounded deterministic policy in environment action coordinates."""

    def __init__(self, settings: ThreeNetworkSettings) -> None:
        super().__init__()
        self.network = _build_mlp(
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


class ThreeNetworkCritic(nn.Module):
    """State-action value network; the two instances never share parameters."""

    def __init__(self, settings: ThreeNetworkSettings) -> None:
        super().__init__()
        self.network = _build_mlp(
            settings.observation_dim + settings.action_dim,
            settings.hidden_layers,
            1,
            activation=settings.activation,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observations, actions), dim=-1))


def shared_residual(
    rewards: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Return delta = r + gamma * mask * Z - Y with explicit masks."""
    if not (
        rewards.shape == bootstrap_mask.shape == z.shape == y.shape
        and rewards.ndim == 2
        and rewards.shape[1] == 1
    ):
        raise ValueError("reward, mask, Z, and Y must share shape (batch, 1)")
    return rewards + gamma * bootstrap_mask * z - y


def shared_squared_error(delta: torch.Tensor) -> torch.Tensor:
    """Return the specified 0.5 * mean(delta ** 2) objective."""
    return 0.5 * delta.square().mean()


@dataclass(frozen=True, slots=True)
class _OutputSnapshot:
    y: torch.Tensor
    y1: torch.Tensor
    z1: torch.Tensor
    z: torch.Tensor
    delta: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PhaseResult:
    loss: float
    gradient_norm: float


def _gradient_norm(gradients: Sequence[torch.Tensor]) -> float:
    if not gradients:
        return 0.0
    return float(
        torch.sqrt(torch.stack([gradient.detach().square().sum() for gradient in gradients]).sum()).cpu()
    )


def _assign_gradients(parameters: Sequence[nn.Parameter], gradients: Sequence[torch.Tensor]) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.detach().clone()


@contextmanager
def _freeze_modules(*modules: nn.Module):
    parameter_states = [
        (parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()
    ]
    try:
        for parameter, _ in parameter_states:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameter_states:
            parameter.requires_grad_(requires_grad)


class ThreeNetworkDDPG(OffPolicyAlgorithm):
    """One actor and two independent critics trained by the shared residual only."""

    capabilities = AlgorithmCapabilities("three_network_ddpg", False, True, True, "experimental")

    def __init__(self, settings: ThreeNetworkSettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        torch.manual_seed(settings.seed)
        self.actor = ThreeNetworkActor(settings).to(self.device)
        self.critic_one = ThreeNetworkCritic(settings).to(self.device)
        self.critic_two = ThreeNetworkCritic(settings).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=settings.actor_learning_rate,
            eps=settings.optimizer_epsilon,
            weight_decay=settings.weight_decay,
        )
        self.critic_one_optimizer = torch.optim.Adam(
            self.critic_one.parameters(),
            lr=settings.critic_one_learning_rate,
            eps=settings.optimizer_epsilon,
            weight_decay=settings.weight_decay,
        )
        self.critic_two_optimizer = torch.optim.Adam(
            self.critic_two.parameters(),
            lr=settings.critic_two_learning_rate,
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
    ) -> "ThreeNetworkDDPG":
        return cls(
            ThreeNetworkSettings.from_config(
                algorithm_config,
                observation_dim=observation_dim,
                action_low=action_low,
                action_high=action_high,
            )
        )

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
        return ActionSelection(
            clip_action(action, self._action_low, self._action_high),
            exploration_applied=exploration,
        )

    def observe_transition(self, transition: Transition) -> None:
        self._environment_steps += 1

    def _zero_all_gradients(self) -> None:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_one_optimizer.zero_grad(set_to_none=True)
        self.critic_two_optimizer.zero_grad(set_to_none=True)

    def _next_actions_detached(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(batch.next_observations)

    def _snapshot(self, batch: TransitionBatch) -> _OutputSnapshot:
        with torch.no_grad():
            next_actions = self.actor(batch.next_observations)
            y = self.critic_one(batch.observations, batch.actions)
            y1 = self.critic_one(batch.next_observations, next_actions)
            z1 = self.critic_two(batch.observations, batch.actions)
            z = self.critic_two(batch.next_observations, next_actions)
            delta = shared_residual(batch.rewards, batch.bootstrap_mask, z, y, self.settings.gamma)
        return _OutputSnapshot(y, y1, z1, z, delta)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        states = (
            batch.observations if self.settings.actor_state_source == "current" else batch.next_observations
        )
        with _freeze_modules(self.critic_one, self.critic_two):
            actions = self.actor(states)
            return -self.critic_two(states, actions).mean()

    def _clip_assigned_gradients(self, parameters: Sequence[nn.Parameter]) -> None:
        if self.settings.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.settings.gradient_clip_norm)

    def _sequential_critic_two(self, batch: TransitionBatch) -> _PhaseResult:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)
        with torch.no_grad():
            y_target = self.critic_one(batch.observations, batch.actions)
        z = self.critic_two(batch.next_observations, next_actions)
        delta = shared_residual(batch.rewards, batch.bootstrap_mask, z, y_target, self.settings.gamma)
        loss = shared_squared_error(delta)
        parameters = tuple(self.critic_two.parameters())
        gradients = torch.autograd.grad(loss, parameters)
        norm = _gradient_norm(gradients)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.critic_two_optimizer.step()
        return _PhaseResult(float(loss.detach().cpu()), norm)

    def _sequential_critic_one(self, batch: TransitionBatch) -> _PhaseResult:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)
        with torch.no_grad():
            z_target = self.critic_two(batch.next_observations, next_actions)
        y = self.critic_one(batch.observations, batch.actions)
        delta = shared_residual(batch.rewards, batch.bootstrap_mask, z_target, y, self.settings.gamma)
        loss = shared_squared_error(delta)
        parameters = tuple(self.critic_one.parameters())
        gradients = torch.autograd.grad(loss, parameters)
        norm = _gradient_norm(gradients)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.critic_one_optimizer.step()
        return _PhaseResult(float(loss.detach().cpu()), norm)

    def _sequential_actor(self, batch: TransitionBatch) -> _PhaseResult:
        self._zero_all_gradients()
        loss = self._actor_loss(batch)
        parameters = tuple(self.actor.parameters())
        gradients = torch.autograd.grad(loss, parameters)
        norm = _gradient_norm(gradients)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.actor_optimizer.step()
        return _PhaseResult(float(loss.detach().cpu()), norm)

    def _sequential_update(self, batch: TransitionBatch) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult]:
        critic_two = self._sequential_critic_two(batch)
        critic_one = self._sequential_critic_one(batch)
        actor = self._sequential_actor(batch)
        return critic_one, critic_two, actor

    def _simultaneous_update(self, batch: TransitionBatch) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult]:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)

        y = self.critic_one(batch.observations, batch.actions)
        z = self.critic_two(batch.next_observations, next_actions)
        critic_two_delta = shared_residual(
            batch.rewards, batch.bootstrap_mask, z, y.detach(), self.settings.gamma
        )
        critic_two_loss = shared_squared_error(critic_two_delta)
        critic_two_parameters = tuple(self.critic_two.parameters())
        critic_two_gradients = torch.autograd.grad(critic_two_loss, critic_two_parameters)

        critic_one_delta = shared_residual(
            batch.rewards, batch.bootstrap_mask, z.detach(), y, self.settings.gamma
        )
        critic_one_loss = shared_squared_error(critic_one_delta)
        critic_one_parameters = tuple(self.critic_one.parameters())
        critic_one_gradients = torch.autograd.grad(critic_one_loss, critic_one_parameters)

        actor_loss = self._actor_loss(batch)
        actor_parameters = tuple(self.actor.parameters())
        actor_gradients = torch.autograd.grad(actor_loss, actor_parameters)

        critic_one_norm = _gradient_norm(critic_one_gradients)
        critic_two_norm = _gradient_norm(critic_two_gradients)
        actor_norm = _gradient_norm(actor_gradients)

        # Store every pre-update gradient before any optimizer mutates parameters.
        _assign_gradients(critic_one_parameters, critic_one_gradients)
        _assign_gradients(critic_two_parameters, critic_two_gradients)
        _assign_gradients(actor_parameters, actor_gradients)
        self._clip_assigned_gradients(critic_one_parameters)
        self._clip_assigned_gradients(critic_two_parameters)
        self._clip_assigned_gradients(actor_parameters)

        self.critic_two_optimizer.step()
        self.critic_one_optimizer.step()
        self.actor_optimizer.step()
        return (
            _PhaseResult(float(critic_one_loss.detach().cpu()), critic_one_norm),
            _PhaseResult(float(critic_two_loss.detach().cpu()), critic_two_norm),
            _PhaseResult(float(actor_loss.detach().cpu()), actor_norm),
        )

    def _actor_residual_interaction(self, batch: TransitionBatch) -> dict[str, float]:
        """Measure actor/shared-error gradient interaction without populating .grad."""
        actor_parameters = tuple(self.actor.parameters())
        with _freeze_modules(self.critic_one, self.critic_two):
            next_actions = self.actor(batch.next_observations)
            y = self.critic_one(batch.observations, batch.actions)
            z = self.critic_two(batch.next_observations, next_actions)
            delta = shared_residual(batch.rewards, batch.bootstrap_mask, z, y, self.settings.gamma)
            residual_loss = shared_squared_error(delta)
            residual_gradients = torch.autograd.grad(residual_loss, actor_parameters)

            actor_states = (
                batch.observations
                if self.settings.actor_state_source == "current"
                else batch.next_observations
            )
            actor_actions = self.actor(actor_states)
            actor_loss = -self.critic_two(actor_states, actor_actions).mean()
            actor_gradients = torch.autograd.grad(actor_loss, actor_parameters)

        inner = sum(
            (residual * actor).sum()
            for residual, actor in zip(residual_gradients, actor_gradients, strict=True)
        )
        residual_norm = torch.sqrt(
            torch.stack([gradient.square().sum() for gradient in residual_gradients]).sum()
        )
        actor_norm = torch.sqrt(torch.stack([gradient.square().sum() for gradient in actor_gradients]).sum())
        denominator = residual_norm * actor_norm
        cosine = torch.where(
            denominator > 0,
            inner / denominator,
            torch.zeros_like(denominator),
        )
        return {
            "actor_shared_gradient_inner_product": float(inner.detach().cpu()),
            "actor_shared_gradient_cosine": float(cosine.detach().cpu()),
        }

    @staticmethod
    def _snapshot_metrics(snapshot: _OutputSnapshot) -> dict[str, float]:
        values: dict[str, float] = {
            "signed_residual_mean": float(snapshot.delta.mean().cpu()),
            "absolute_residual_mean": float(snapshot.delta.abs().mean().cpu()),
            "shared_squared_error": float(shared_squared_error(snapshot.delta).cpu()),
        }
        outputs = {
            "y": snapshot.y,
            "y1": snapshot.y1,
            "z1": snapshot.z1,
            "z": snapshot.z,
        }
        for name, tensor in outputs.items():
            values[f"{name}_mean"] = float(tensor.mean().cpu())
            values[f"{name}_std"] = float(tensor.std(unbiased=False).cpu())
            values[f"{name}_abs_mean"] = float(tensor.abs().mean().cpu())
            values[f"{name}_abs_max"] = float(tensor.abs().max().cpu())
        combined = torch.cat(tuple(outputs.values()), dim=0)
        values["q_abs_mean"] = float(combined.abs().mean().cpu())
        values["q_abs_max"] = float(combined.abs().max().cpu())
        return values

    def update_from_batch(self, batch: TransitionBatch) -> UpdateResult:
        batch = batch.to(self.device)
        next_step = self._update_step + 1
        snapshot = self._snapshot(batch)
        diagnostic_values: dict[str, float] = {}
        if (
            self.settings.diagnostic_frequency is not None
            and next_step % self.settings.diagnostic_frequency == 0
        ):
            diagnostic_values = self._actor_residual_interaction(batch)

        if self.settings.update_schedule == "sequential":
            critic_one, critic_two, actor = self._sequential_update(batch)
        else:
            critic_one, critic_two, actor = self._simultaneous_update(batch)

        self._update_step = next_step
        values = self._snapshot_metrics(snapshot)
        values.update(
            {
                "critic_one_loss": critic_one.loss,
                "critic_two_loss": critic_two.loss,
                "actor_loss": actor.loss,
                "actor_objective": -actor.loss,
                "critic_one_gradient_norm": critic_one.gradient_norm,
                "critic_two_gradient_norm": critic_two.gradient_norm,
                "actor_gradient_norm": actor.gradient_norm,
                "environment_steps": float(self._environment_steps),
                "update_steps": float(next_step),
            }
        )
        values.update(diagnostic_values)
        return UpdateResult(
            metrics=tuple(MetricRecord(name, value, next_step, "train") for name, value in values.items()),
            updates_applied=1,
        )

    def set_evaluation_mode(self, enabled: bool) -> None:
        self._evaluation_mode = enabled
        self.actor.train(not enabled)
        self.critic_one.train(not enabled)
        self.critic_two.train(not enabled)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "actor": self.actor.state_dict(),
            "critic_one": self.critic_one.state_dict(),
            "critic_two": self.critic_two.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_one_optimizer": self.critic_one_optimizer.state_dict(),
            "critic_two_optimizer": self.critic_two_optimizer.state_dict(),
            "update_step": self._update_step,
            "environment_steps": self._environment_steps,
            "noise_rng_state": self._noise_rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("settings") != self.settings.to_dict():
            raise ValueError("checkpoint settings do not match this three-network instance")
        forbidden = {"target_actor", "target_critic"} & set(state)
        if forbidden:
            raise ValueError("target-network state is forbidden in the proposed variant")
        self.actor.load_state_dict(state["actor"])
        self.critic_one.load_state_dict(state["critic_one"])
        self.critic_two.load_state_dict(state["critic_two"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_one_optimizer.load_state_dict(state["critic_one_optimizer"])
        self.critic_two_optimizer.load_state_dict(state["critic_two_optimizer"])
        self._update_step = int(state["update_step"])
        self._environment_steps = int(state["environment_steps"])
        self._noise_rng.bit_generator.state = state["noise_rng_state"]

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(self.state_dict()), path)

    def restore_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state)
