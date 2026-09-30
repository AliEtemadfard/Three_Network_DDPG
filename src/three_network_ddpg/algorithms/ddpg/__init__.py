"""Verified standard DDPG baseline."""

from .implementation import (
    DDPG,
    DDPGNoTargetActor,
    DDPGSettings,
    DeterministicActor,
    QCritic,
    build_mlp,
    ddpg_td_target,
    soft_update,
)

__all__ = [
    "DDPG",
    "DDPGSettings",
    "DDPGNoTargetActor",
    "DeterministicActor",
    "QCritic",
    "build_mlp",
    "ddpg_td_target",
    "soft_update",
]
