"""Resolve user-facing device requests at the config-to-runtime boundary."""

from typing import Literal

import torch

ResolvedDevice = Literal["cpu", "cuda"]


def resolve_device(requested: str) -> ResolvedDevice:
    """Resolve ``auto`` once while preserving explicit CPU/CUDA requests."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested in {"cpu", "cuda"}:
        return requested
    raise ValueError("device must be cpu, cuda, or auto")
