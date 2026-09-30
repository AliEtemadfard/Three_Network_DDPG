"""Continuous-action transforms for finite Gymnasium Box bounds."""

import numpy as np
import torch


def _checked(
    action: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(action, dtype=np.float32)
    lower = np.asarray(low, dtype=np.float32)
    upper = np.asarray(high, dtype=np.float32)
    if values.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError("action and bound shapes must match exactly")
    if not all(np.all(np.isfinite(array)) for array in (values, lower, upper)):
        raise ValueError("action bounds and values must be finite")
    if np.any(lower >= upper):
        raise ValueError("every action lower bound must be below its upper bound")
    return values, lower, upper


def scale_action(normalized: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map normalized [-1, 1] coordinates to environment bounds without clipping."""
    values, lower, upper = _checked(normalized, low, high)
    return (lower + (values + 1.0) * (upper - lower) / 2.0).astype(np.float32)


def scale_action_tensor(normalized: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Differentiably map normalized actions to finite environment bounds."""
    if normalized.shape[-1] != low.numel() or low.shape != high.shape:
        raise ValueError("last action dimension and bound shapes must match")
    if not torch.all(torch.isfinite(low)) or not torch.all(torch.isfinite(high)):
        raise ValueError("action bounds must be finite")
    if torch.any(low >= high):
        raise ValueError("every action lower bound must be below its upper bound")
    return low + (normalized + 1.0) * (high - low) / 2.0


def unscale_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Invert scale_action, including values outside the nominal action bounds."""
    values, lower, upper = _checked(action, low, high)
    return (2.0 * (values - lower) / (upper - lower) - 1.0).astype(np.float32)


def clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Explicitly clip an environment-scale action to its Box bounds."""
    values, lower, upper = _checked(action, low, high)
    return np.clip(values, lower, upper).astype(np.float32)
