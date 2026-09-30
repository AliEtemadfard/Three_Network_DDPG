"""Shared compatibility checks for resumable algorithm checkpoints."""

from collections.abc import Mapping, Sequence
from typing import Any


def _normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_normalized(item) for item in value)
    return value


def validate_checkpoint_settings(
    saved: Any,
    current: Mapping[str, Any],
    *,
    algorithm_name: str,
) -> None:
    """Reject training-semantic differences while allowing runtime relocation."""
    if not isinstance(saved, Mapping):
        raise ValueError(f"{algorithm_name} checkpoint is missing settings")
    ignored = {"device", "seed"}
    saved_semantics = {key: value for key, value in saved.items() if key not in ignored}
    current_semantics = {key: value for key, value in current.items() if key not in ignored}
    if _normalized(saved_semantics) != _normalized(current_semantics):
        raise ValueError(f"checkpoint settings do not match this {algorithm_name} instance")


def validate_adam_optimizer_state(
    state: Any,
    *,
    learning_rate: float,
    epsilon: float,
    weight_decay: float,
    optimizer_name: str,
) -> None:
    """Ensure serialized Adam parameter groups match the active settings."""
    if not isinstance(state, Mapping):
        raise ValueError(f"{optimizer_name} checkpoint state is invalid")
    groups = state.get("param_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, str | bytes) or not groups:
        raise ValueError(f"{optimizer_name} checkpoint parameter groups are invalid")
    expected = {
        "lr": float(learning_rate),
        "betas": (0.9, 0.999),
        "eps": float(epsilon),
        "weight_decay": float(weight_decay),
        "amsgrad": False,
        "maximize": False,
    }
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError(f"{optimizer_name} checkpoint parameter group is invalid")
        for key, value in expected.items():
            if key not in group or _normalized(group[key]) != _normalized(value):
                label = "learning rate (lr)" if key == "lr" else key
                raise ValueError(f"{optimizer_name} checkpoint {label} does not match current settings")
