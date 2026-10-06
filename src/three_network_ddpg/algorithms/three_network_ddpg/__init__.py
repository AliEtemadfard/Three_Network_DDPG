"""Three-network shared-error algorithm and controlled ablations."""

from .implementation import (
    ThreeNetworkActor,
    ThreeNetworkCritic,
    ThreeNetworkDDPG,
    ThreeNetworkSettings,
    agreement_error,
    beta_regularizer,
    q2_td_residual,
    shared_residual,
    shared_squared_error,
)

__all__ = [
    "ThreeNetworkActor",
    "ThreeNetworkCritic",
    "ThreeNetworkDDPG",
    "ThreeNetworkSettings",
    "agreement_error",
    "beta_regularizer",
    "q2_td_residual",
    "shared_residual",
    "shared_squared_error",
]
