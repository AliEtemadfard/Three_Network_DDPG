"""Target-free three-network actor-critic with controlled research ablations."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from ...core.actions import clip_action, scale_action_tensor
from ...core.batches import Transition, TransitionBatch
from ...core.interfaces import (
    ActionSelection,
    AlgorithmCapabilities,
    MetricRecord,
    OffPolicyAlgorithm,
    UpdateResult,
)

AgreementMode = Literal["none", "current", "next", "both"]
BetaMode = Literal["none", "linear_y", "l2_y", "l2_y_z"]
UpdateSchedule = Literal["sequential", "simultaneous", "guarded"]
GuardScope = Literal["joint", "critics", "actor"]
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
    agreement_mode: AgreementMode = "none"
    alpha: float = 0.0
    alpha_prime: float = 0.0
    beta_mode: BetaMode = "none"
    beta_coefficient: float = 0.0
    update_schedule: UpdateSchedule = "sequential"
    actor_state_source: ActorStateSource = "next"
    guard_scope: GuardScope = "joint"
    guard_tolerance: float = 0.0
    guard_backtracking_factor: float = 0.5
    guard_maximum_retries: int = 5
    guard_minimum_learning_rate: float = 1e-6
    guard_validation_batch: bool = False
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
        learning_rates = (
            self.actor_learning_rate,
            self.critic_one_learning_rate,
            self.critic_two_learning_rate,
        )
        if any(rate <= 0 for rate in learning_rates):
            raise ValueError("actor and critic learning rates must be positive")
        if self.optimizer_epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer epsilon must be positive and weight decay nonnegative")
        if self.agreement_mode not in {"none", "current", "next", "both"}:
            raise ValueError("unsupported agreement_mode")
        if self.alpha < 0 or self.alpha_prime < 0:
            raise ValueError("alpha and alpha_prime must be nonnegative")
        if self.agreement_mode == "none" and (self.alpha != 0 or self.alpha_prime != 0):
            raise ValueError("agreement mode none requires zero alpha coefficients")
        if self.agreement_mode == "current" and self.alpha != 0:
            raise ValueError("current agreement does not use alpha")
        if self.agreement_mode == "next" and self.alpha_prime != 0:
            raise ValueError("next agreement does not use alpha_prime")
        if self.beta_mode not in {"none", "linear_y", "l2_y", "l2_y_z"}:
            raise ValueError("unsupported beta_mode")
        if self.beta_coefficient < 0:
            raise ValueError("beta_coefficient must be nonnegative")
        if self.beta_mode == "none" and self.beta_coefficient != 0:
            raise ValueError("beta mode none requires a zero coefficient")
        if self.beta_mode != "none" and self.beta_coefficient == 0:
            raise ValueError("an active beta mode requires a positive coefficient")
        if self.update_schedule not in {"sequential", "simultaneous", "guarded"}:
            raise ValueError("unsupported update_schedule")
        if self.actor_state_source not in {"current", "next"}:
            raise ValueError("actor_state_source must be current or next")
        if self.guard_scope not in {"joint", "critics", "actor"}:
            raise ValueError("guard_scope must be joint, critics, or actor")
        if self.guard_tolerance < 0:
            raise ValueError("guard_tolerance must be nonnegative")
        if not 0 < self.guard_backtracking_factor < 1:
            raise ValueError("guard_backtracking_factor must be in (0, 1)")
        if self.guard_maximum_retries < 0:
            raise ValueError("guard_maximum_retries must be nonnegative")
        if self.guard_minimum_learning_rate <= 0:
            raise ValueError("guard_minimum_learning_rate must be positive")
        if self.guard_minimum_learning_rate > min(learning_rates):
            raise ValueError("guard_minimum_learning_rate exceeds a configured learning rate")
        if not isinstance(self.guard_validation_batch, bool):
            raise TypeError("guard_validation_batch must be boolean")
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
        if algorithm_config["critic_initialization"] != "independent":
            raise ValueError("three-network ablations require independent critic initialization")
        if int(algorithm_config["actor_update_delay"]) != 1:
            raise ValueError("three-network ablations update the actor on every learner step")
        if algorithm_config["optimizer"]["name"] != "adam":
            raise ValueError("three-network ablations currently support Adam")
        if not bool(algorithm_config["action_rescaling"]):
            raise ValueError("three-network ablations require action rescaling")
        noise = algorithm_config["exploration_noise"]
        if noise["mode"] not in {"gaussian", "none"}:
            raise ValueError("three-network ablations support gaussian or disabled exploration")
        beta = algorithm_config["beta_regularizer"]
        guard = algorithm_config["guard"]
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
            agreement_mode=str(algorithm_config["agreement_mode"]),
            alpha=float(algorithm_config["alpha"]),
            alpha_prime=float(algorithm_config["alpha_prime"]),
            beta_mode=str(beta["mode"]),
            beta_coefficient=float(beta["coefficient"]),
            update_schedule=str(algorithm_config["update_schedule"]),
            actor_state_source=str(algorithm_config["actor_state_source"]),
            guard_scope=str(guard["scope"]),
            guard_tolerance=float(guard["tolerance"]),
            guard_backtracking_factor=float(guard["backtracking_factor"]),
            guard_maximum_retries=int(guard["maximum_retries"]),
            guard_minimum_learning_rate=float(guard["minimum_learning_rate"]),
            guard_validation_batch=bool(guard["validation_batch"]),
            exploration_noise_std=(0.0 if noise["mode"] == "none" else float(noise["standard_deviation"])),
            exploration_noise_clip=None if noise["clip"] is None else float(noise["clip"]),
            gradient_clip_norm=None if gradient_clip is None else float(gradient_clip),
            diagnostic_frequency=(None if diagnostic_frequency is None else int(diagnostic_frequency)),
            device=str(algorithm_config["device"]),
            seed=int(algorithm_config["seed"]),
        )

    def to_dict(self) -> dict[str, Any]:
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
    if not (
        rewards.shape == bootstrap_mask.shape == z.shape == y.shape
        and rewards.ndim == 2
        and rewards.shape[1] == 1
    ):
        raise ValueError("reward, mask, Z, and Y must share shape (batch, 1)")
    return rewards + gamma * bootstrap_mask * z - y


def shared_squared_error(delta: torch.Tensor) -> torch.Tensor:
    return 0.5 * delta.square().mean()


def agreement_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("agreement tensors must have matching shapes")
    return 0.5 * (left - right).square().mean()


def beta_regularizer(mode: BetaMode, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the unweighted beta term; linear_y is experimental and unbounded."""
    if y.shape != z.shape:
        raise ValueError("Y and Z must have matching shapes")
    if mode == "none":
        return y.new_zeros(())
    if mode == "linear_y":
        return y.mean()
    if mode == "l2_y":
        return 0.5 * y.square().mean()
    if mode == "l2_y_z":
        return 0.5 * (y.square() + z.square()).mean()
    raise ValueError(f"unsupported beta mode: {mode}")


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
    shared_loss: float
    agreement_loss: float
    beta_loss: float
    gradient_norm: float


@dataclass(frozen=True, slots=True)
class _GuardOutcome:
    accepted: bool
    retries: int
    effective_learning_rate: float
    learning_rate_scale: float
    error_before: float
    error_after: float
    committed_error_after: float
    validation_change: float | None
    critic_one: _PhaseResult | None
    critic_two: _PhaseResult | None
    actor: _PhaseResult | None
    update_applied: bool


def _gradient_norm(gradients: Sequence[torch.Tensor]) -> float:
    if not gradients:
        return 0.0
    total = torch.stack([gradient.detach().square().sum() for gradient in gradients]).sum()
    return float(torch.sqrt(total).cpu())


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
    """Three trainable networks with explicit ablation and guard semantics."""

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
        self._guard_accepted_steps = 0
        self._guard_rejected_steps = 0
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

    @property
    def requires_validation_batch(self) -> bool:
        return self.settings.update_schedule == "guarded" and self.settings.guard_validation_batch

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

    def _shared_error_value(self, batch: TransitionBatch) -> float:
        return float(shared_squared_error(self._snapshot(batch).delta).cpu())

    def _uses_current_agreement(self) -> bool:
        return self.settings.agreement_mode in {"current", "both"}

    def _uses_next_agreement(self) -> bool:
        return self.settings.agreement_mode in {"next", "both"}

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

    @staticmethod
    def _phase_result(
        total: torch.Tensor,
        shared: torch.Tensor,
        agreement: torch.Tensor,
        beta: torch.Tensor,
        gradients: Sequence[torch.Tensor],
    ) -> _PhaseResult:
        return _PhaseResult(
            loss=float(total.detach().cpu()),
            shared_loss=float(shared.detach().cpu()),
            agreement_loss=float(agreement.detach().cpu()),
            beta_loss=float(beta.detach().cpu()),
            gradient_norm=_gradient_norm(gradients),
        )

    def _sequential_critic_two(self, batch: TransitionBatch) -> _PhaseResult:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)
        with torch.no_grad():
            y_target = self.critic_one(batch.observations, batch.actions)
        z = self.critic_two(batch.next_observations, next_actions)
        z1 = self.critic_two(batch.observations, batch.actions)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z, y_target, self.settings.gamma)
        )
        current = agreement_error(z1, y_target) if self._uses_current_agreement() else shared.new_zeros(())
        total = shared + self.settings.alpha_prime * current
        parameters = tuple(self.critic_two.parameters())
        gradients = torch.autograd.grad(total, parameters)
        result = self._phase_result(total, shared, current, shared.new_zeros(()), gradients)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.critic_two_optimizer.step()
        return result

    def _sequential_critic_one(self, batch: TransitionBatch) -> _PhaseResult:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)
        with torch.no_grad():
            z_target = self.critic_two(batch.next_observations, next_actions)
        y = self.critic_one(batch.observations, batch.actions)
        y1 = self.critic_one(batch.next_observations, next_actions)
        shared = shared_squared_error(
            shared_residual(batch.rewards, batch.bootstrap_mask, z_target, y, self.settings.gamma)
        )
        next_agreement = (
            agreement_error(y1, z_target) if self._uses_next_agreement() else shared.new_zeros(())
        )
        beta = beta_regularizer(self.settings.beta_mode, y, z_target)
        total = shared + self.settings.alpha * next_agreement + self.settings.beta_coefficient * beta
        parameters = tuple(self.critic_one.parameters())
        gradients = torch.autograd.grad(total, parameters)
        result = self._phase_result(total, shared, next_agreement, beta, gradients)
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.critic_one_optimizer.step()
        return result

    def _sequential_actor(self, batch: TransitionBatch) -> _PhaseResult:
        self._zero_all_gradients()
        loss = self._actor_loss(batch)
        parameters = tuple(self.actor.parameters())
        gradients = torch.autograd.grad(loss, parameters)
        result = self._phase_result(
            loss, loss.new_zeros(()), loss.new_zeros(()), loss.new_zeros(()), gradients
        )
        _assign_gradients(parameters, gradients)
        self._clip_assigned_gradients(parameters)
        self.actor_optimizer.step()
        return result

    def _sequential_update(self, batch: TransitionBatch) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult]:
        critic_two = self._sequential_critic_two(batch)
        critic_one = self._sequential_critic_one(batch)
        actor = self._sequential_actor(batch)
        return critic_one, critic_two, actor

    def _simultaneous_update(self, batch: TransitionBatch) -> tuple[_PhaseResult, _PhaseResult, _PhaseResult]:
        self._zero_all_gradients()
        next_actions = self._next_actions_detached(batch)
        y = self.critic_one(batch.observations, batch.actions)
        y1 = self.critic_one(batch.next_observations, next_actions)
        z1 = self.critic_two(batch.observations, batch.actions)
        z = self.critic_two(batch.next_observations, next_actions)

        critic_two_shared = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z,
                y.detach(),
                self.settings.gamma,
            )
        )
        current = (
            agreement_error(z1, y.detach())
            if self._uses_current_agreement()
            else critic_two_shared.new_zeros(())
        )
        critic_two_loss = critic_two_shared + self.settings.alpha_prime * current
        critic_two_parameters = tuple(self.critic_two.parameters())
        critic_two_gradients = torch.autograd.grad(critic_two_loss, critic_two_parameters)

        critic_one_shared = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z.detach(),
                y,
                self.settings.gamma,
            )
        )
        next_agreement = (
            agreement_error(y1, z.detach())
            if self._uses_next_agreement()
            else critic_one_shared.new_zeros(())
        )
        beta = beta_regularizer(self.settings.beta_mode, y, z.detach())
        critic_one_loss = (
            critic_one_shared + self.settings.alpha * next_agreement + self.settings.beta_coefficient * beta
        )
        critic_one_parameters = tuple(self.critic_one.parameters())
        critic_one_gradients = torch.autograd.grad(critic_one_loss, critic_one_parameters)

        actor_loss = self._actor_loss(batch)
        actor_parameters = tuple(self.actor.parameters())
        actor_gradients = torch.autograd.grad(actor_loss, actor_parameters)

        critic_one_result = self._phase_result(
            critic_one_loss,
            critic_one_shared,
            next_agreement,
            beta,
            critic_one_gradients,
        )
        critic_two_result = self._phase_result(
            critic_two_loss,
            critic_two_shared,
            current,
            critic_two_shared.new_zeros(()),
            critic_two_gradients,
        )
        actor_result = self._phase_result(
            actor_loss,
            actor_loss.new_zeros(()),
            actor_loss.new_zeros(()),
            actor_loss.new_zeros(()),
            actor_gradients,
        )

        _assign_gradients(critic_one_parameters, critic_one_gradients)
        _assign_gradients(critic_two_parameters, critic_two_gradients)
        _assign_gradients(actor_parameters, actor_gradients)
        self._clip_assigned_gradients(critic_one_parameters)
        self._clip_assigned_gradients(critic_two_parameters)
        self._clip_assigned_gradients(actor_parameters)
        self.critic_two_optimizer.step()
        self.critic_one_optimizer.step()
        self.actor_optimizer.step()
        return critic_one_result, critic_two_result, actor_result

    def _capture_training_state(self) -> dict[str, Any]:
        return deepcopy(
            {
                "actor": self.actor.state_dict(),
                "critic_one": self.critic_one.state_dict(),
                "critic_two": self.critic_two.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_one_optimizer": self.critic_one_optimizer.state_dict(),
                "critic_two_optimizer": self.critic_two_optimizer.state_dict(),
            }
        )

    def _restore_training_state(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(deepcopy(state["actor"]))
        self.critic_one.load_state_dict(deepcopy(state["critic_one"]))
        self.critic_two.load_state_dict(deepcopy(state["critic_two"]))
        # load_state_dict may retain references to optimizer-state tensors. Copy
        # every load so a retry cannot mutate the saved guard snapshot.
        self.actor_optimizer.load_state_dict(deepcopy(state["actor_optimizer"]))
        self.critic_one_optimizer.load_state_dict(deepcopy(state["critic_one_optimizer"]))
        self.critic_two_optimizer.load_state_dict(deepcopy(state["critic_two_optimizer"]))
        self._zero_all_gradients()

    def _guard_optimizers(self, scope: GuardScope) -> tuple[Optimizer, ...]:
        if scope == "joint":
            return (
                self.critic_two_optimizer,
                self.critic_one_optimizer,
                self.actor_optimizer,
            )
        if scope == "critics":
            return self.critic_two_optimizer, self.critic_one_optimizer
        return (self.actor_optimizer,)

    @staticmethod
    def _optimizer_lrs(optimizers: Sequence[Optimizer]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(group["lr"]) for group in optimizer.param_groups) for optimizer in optimizers
        )

    @staticmethod
    def _set_scaled_lrs(
        optimizers: Sequence[Optimizer],
        base_lrs: Sequence[Sequence[float]],
        scale: float,
    ) -> float:
        effective: list[float] = []
        for optimizer, group_lrs in zip(optimizers, base_lrs, strict=True):
            for group, base_lr in zip(optimizer.param_groups, group_lrs, strict=True):
                group["lr"] = base_lr * scale
                effective.append(float(group["lr"]))
        return min(effective)

    @staticmethod
    def _restore_lrs(optimizers: Sequence[Optimizer], base_lrs: Sequence[Sequence[float]]) -> None:
        for optimizer, group_lrs in zip(optimizers, base_lrs, strict=True):
            for group, base_lr in zip(optimizer.param_groups, group_lrs, strict=True):
                group["lr"] = base_lr

    def _guard_proposal(
        self, scope: GuardScope, batch: TransitionBatch
    ) -> tuple[_PhaseResult | None, _PhaseResult | None, _PhaseResult | None]:
        if scope == "joint":
            return self._sequential_update(batch)
        if scope == "critics":
            critic_two = self._sequential_critic_two(batch)
            critic_one = self._sequential_critic_one(batch)
            return critic_one, critic_two, None
        return None, None, self._sequential_actor(batch)

    def _run_guard(
        self,
        scope: GuardScope,
        batch: TransitionBatch,
        validation_batch: TransitionBatch | None,
    ) -> _GuardOutcome:
        state = self._capture_training_state()
        optimizers = self._guard_optimizers(scope)
        base_lrs = self._optimizer_lrs(optimizers)
        error_before = self._shared_error_value(batch)
        validation_before = None if validation_batch is None else self._shared_error_value(validation_batch)
        accepted = False
        retries = 0
        effective_lr = min(rate for group in base_lrs for rate in group)
        scale = 1.0
        error_after = error_before
        validation_change: float | None = None
        phase_results: tuple[_PhaseResult | None, _PhaseResult | None, _PhaseResult | None] = (
            None,
            None,
            None,
        )

        for retry in range(self.settings.guard_maximum_retries + 1):
            scale = self.settings.guard_backtracking_factor**retry
            candidate_lr = min(rate * scale for group in base_lrs for rate in group)
            if candidate_lr < self.settings.guard_minimum_learning_rate:
                break
            self._restore_training_state(state)
            effective_lr = self._set_scaled_lrs(optimizers, base_lrs, scale)
            phase_results = self._guard_proposal(scope, batch)
            error_after = self._shared_error_value(batch)
            if validation_batch is not None and validation_before is not None:
                validation_change = self._shared_error_value(validation_batch) - validation_before
            if error_after <= error_before + self.settings.guard_tolerance:
                accepted = True
                retries = retry
                break
            retries = retry

        if accepted:
            # The accepted proposal used the reported effective rate. Restore the
            # configured rates so backtracking does not compound across updates.
            self._restore_lrs(optimizers, base_lrs)
            self._guard_accepted_steps += 1
            committed_error_after = error_after
        else:
            self._restore_training_state(state)
            self._guard_rejected_steps += 1
            committed_error_after = self._shared_error_value(batch)

        return _GuardOutcome(
            accepted=accepted,
            retries=retries,
            effective_learning_rate=effective_lr,
            learning_rate_scale=scale,
            error_before=error_before,
            error_after=error_after,
            committed_error_after=committed_error_after,
            validation_change=validation_change,
            critic_one=phase_results[0],
            critic_two=phase_results[1],
            actor=phase_results[2],
            update_applied=accepted,
        )

    def _guarded_update(
        self,
        batch: TransitionBatch,
        validation_batch: TransitionBatch | None,
    ) -> _GuardOutcome:
        scope = self.settings.guard_scope
        if scope == "actor":
            critic_two = self._sequential_critic_two(batch)
            critic_one = self._sequential_critic_one(batch)
            guarded = self._run_guard("actor", batch, validation_batch)
            return replace(
                guarded,
                critic_one=critic_one,
                critic_two=critic_two,
                update_applied=True,
            )
        guarded = self._run_guard(scope, batch, validation_batch)
        if scope == "critics":
            actor = self._sequential_actor(batch)
            return replace(guarded, actor=actor, update_applied=True)
        return guarded

    def _gradient_diagnostics(self, batch: TransitionBatch) -> dict[str, float]:
        actor_parameters = tuple(self.actor.parameters())
        critic_one_parameters = tuple(self.critic_one.parameters())
        critic_two_parameters = tuple(self.critic_two.parameters())
        next_actions = self._next_actions_detached(batch)

        y = self.critic_one(batch.observations, batch.actions)
        y1 = self.critic_one(batch.next_observations, next_actions)
        with torch.no_grad():
            z_target = self.critic_two(batch.next_observations, next_actions)
        critic_one_shared = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z_target,
                y,
                self.settings.gamma,
            )
        )
        next_agreement = (
            agreement_error(y1, z_target) if self._uses_next_agreement() else critic_one_shared.new_zeros(())
        )
        beta = beta_regularizer(self.settings.beta_mode, y, z_target)
        critic_one_objective = (
            critic_one_shared + self.settings.alpha * next_agreement + self.settings.beta_coefficient * beta
        )
        one_shared_gradients = torch.autograd.grad(
            critic_one_shared, critic_one_parameters, retain_graph=True
        )
        one_objective_gradients = torch.autograd.grad(critic_one_objective, critic_one_parameters)

        z = self.critic_two(batch.next_observations, next_actions)
        z1 = self.critic_two(batch.observations, batch.actions)
        with torch.no_grad():
            y_target = self.critic_one(batch.observations, batch.actions)
        critic_two_shared = shared_squared_error(
            shared_residual(
                batch.rewards,
                batch.bootstrap_mask,
                z,
                y_target,
                self.settings.gamma,
            )
        )
        current = (
            agreement_error(z1, y_target)
            if self._uses_current_agreement()
            else critic_two_shared.new_zeros(())
        )
        critic_two_objective = critic_two_shared + self.settings.alpha_prime * current
        two_shared_gradients = torch.autograd.grad(
            critic_two_shared, critic_two_parameters, retain_graph=True
        )
        two_objective_gradients = torch.autograd.grad(critic_two_objective, critic_two_parameters)

        with _freeze_modules(self.critic_one, self.critic_two):
            actor_next_actions = self.actor(batch.next_observations)
            y_fixed = self.critic_one(batch.observations, batch.actions)
            z_actor = self.critic_two(batch.next_observations, actor_next_actions)
            actor_shared = shared_squared_error(
                shared_residual(
                    batch.rewards,
                    batch.bootstrap_mask,
                    z_actor,
                    y_fixed,
                    self.settings.gamma,
                )
            )
            actor_shared_gradients = torch.autograd.grad(actor_shared, actor_parameters)
            actor_states = (
                batch.observations
                if self.settings.actor_state_source == "current"
                else batch.next_observations
            )
            actor_loss = -self.critic_two(actor_states, self.actor(actor_states)).mean()
            actor_gradients = torch.autograd.grad(actor_loss, actor_parameters)

        values: dict[str, float] = {}
        for name, shared_gradients, objective_gradients in (
            ("critic_one", one_shared_gradients, one_objective_gradients),
            ("critic_two", two_shared_gradients, two_objective_gradients),
            ("actor", actor_shared_gradients, actor_gradients),
        ):
            inner = sum(
                (shared_gradient * objective_gradient).sum()
                for shared_gradient, objective_gradient in zip(
                    shared_gradients, objective_gradients, strict=True
                )
            )
            shared_norm = torch.sqrt(
                torch.stack([gradient.square().sum() for gradient in shared_gradients]).sum()
            )
            objective_norm = torch.sqrt(
                torch.stack([gradient.square().sum() for gradient in objective_gradients]).sum()
            )
            denominator = shared_norm * objective_norm
            cosine = torch.where(
                denominator > 0,
                inner / denominator,
                torch.zeros_like(denominator),
            )
            values[f"{name}_shared_gradient_inner_product"] = float(inner.detach().cpu())
            values[f"{name}_shared_gradient_cosine"] = float(cosine.detach().cpu())
        return values

    def _snapshot_metrics(self, snapshot: _OutputSnapshot) -> dict[str, float]:
        values: dict[str, float] = {
            "signed_residual_mean": float(snapshot.delta.mean().cpu()),
            "absolute_residual_mean": float(snapshot.delta.abs().mean().cpu()),
            "shared_squared_error": float(shared_squared_error(snapshot.delta).cpu()),
            "current_agreement_error": float(agreement_error(snapshot.z1, snapshot.y).cpu()),
            "next_agreement_error": float(agreement_error(snapshot.y1, snapshot.z).cpu()),
            "beta_regularizer_value": float(
                beta_regularizer(self.settings.beta_mode, snapshot.y, snapshot.z).cpu()
            ),
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

    @staticmethod
    def _add_phase_metrics(values: dict[str, float], name: str, result: _PhaseResult | None) -> None:
        if result is None:
            return
        values[f"{name}_loss"] = result.loss
        values[f"{name}_shared_loss"] = result.shared_loss
        values[f"{name}_agreement_loss"] = result.agreement_loss
        values[f"{name}_beta_loss"] = result.beta_loss
        values[f"{name}_gradient_norm"] = result.gradient_norm

    def update_from_batch(
        self,
        batch: TransitionBatch,
        *,
        validation_batch: TransitionBatch | None = None,
    ) -> UpdateResult:
        batch = batch.to(self.device)
        validation = None if validation_batch is None else validation_batch.to(self.device)
        if self.requires_validation_batch and validation is None:
            raise ValueError("guarded validation monitoring requires a validation batch")
        next_step = self._update_step + 1
        snapshot = self._snapshot(batch)
        diagnostic_values: dict[str, float] = {}
        if (
            self.settings.diagnostic_frequency is not None
            and next_step % self.settings.diagnostic_frequency == 0
        ):
            diagnostic_values = self._gradient_diagnostics(batch)

        guard_outcome: _GuardOutcome | None = None
        if self.settings.update_schedule == "sequential":
            critic_one, critic_two, actor = self._sequential_update(batch)
            updates_applied = 1
        elif self.settings.update_schedule == "simultaneous":
            critic_one, critic_two, actor = self._simultaneous_update(batch)
            updates_applied = 1
        else:
            guard_outcome = self._guarded_update(batch, validation)
            critic_one = guard_outcome.critic_one
            critic_two = guard_outcome.critic_two
            actor = guard_outcome.actor
            updates_applied = int(guard_outcome.update_applied)

        self._update_step = next_step
        values = self._snapshot_metrics(snapshot)
        self._add_phase_metrics(values, "critic_one", critic_one)
        self._add_phase_metrics(values, "critic_two", critic_two)
        self._add_phase_metrics(values, "actor", actor)
        if actor is not None:
            values["actor_objective"] = -actor.loss
        values.update(
            {
                "environment_steps": float(self._environment_steps),
                "update_steps": float(next_step),
            }
        )
        if guard_outcome is not None:
            values.update(
                {
                    "guard_accepted_step": float(guard_outcome.accepted),
                    "guard_rejected_step": float(not guard_outcome.accepted),
                    "guard_accepted_steps": float(self._guard_accepted_steps),
                    "guard_rejected_steps": float(self._guard_rejected_steps),
                    "guard_retry_count": float(guard_outcome.retries),
                    "guard_effective_learning_rate": guard_outcome.effective_learning_rate,
                    "guard_learning_rate_scale": guard_outcome.learning_rate_scale,
                    "guard_error_before": guard_outcome.error_before,
                    "guard_error_after": guard_outcome.error_after,
                    "guard_committed_error_after": guard_outcome.committed_error_after,
                }
            )
            if guard_outcome.validation_change is not None:
                values["guard_validation_error_change"] = guard_outcome.validation_change
        values.update(diagnostic_values)
        skipped_reason = (
            "guard rejected joint proposal"
            if guard_outcome is not None and not guard_outcome.update_applied
            else None
        )
        return UpdateResult(
            metrics=tuple(MetricRecord(name, value, next_step, "train") for name, value in values.items()),
            updates_applied=updates_applied,
            skipped_reason=skipped_reason,
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
            "guard_accepted_steps": self._guard_accepted_steps,
            "guard_rejected_steps": self._guard_rejected_steps,
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
        self._guard_accepted_steps = int(state.get("guard_accepted_steps", 0))
        self._guard_rejected_steps = int(state.get("guard_rejected_steps", 0))
        self._noise_rng.bit_generator.state = state["noise_rng_state"]

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(self.state_dict()), path)

    def restore_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state)
