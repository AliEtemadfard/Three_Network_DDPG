"""Minimal shared-error three-network algorithm."""

from .implementation import (
    ThreeNetworkActor,
    ThreeNetworkCritic,
    ThreeNetworkDDPG,
    ThreeNetworkSettings,
    shared_residual,
    shared_squared_error,
)

__all__ = [
    "ThreeNetworkActor",
    "ThreeNetworkCritic",
    "ThreeNetworkDDPG",
    "ThreeNetworkSettings",
    "shared_residual",
    "shared_squared_error",
]
