"""Algorithm capability registry and reviewed construction entry point."""

from typing import Any

from ..core.interfaces import AlgorithmCapabilities

ALGORITHMS: dict[str, AlgorithmCapabilities] = {
    "ddpg": AlgorithmCapabilities("ddpg", False, True, True, "verified"),
    "ddpg_no_target_actor": AlgorithmCapabilities("ddpg_no_target_actor", False, True, True, "experimental"),
    "three_network_ddpg": AlgorithmCapabilities("three_network_ddpg", False, True, True, "experimental"),
    "three_network_shared_d": AlgorithmCapabilities(
        "three_network_shared_d", False, True, True, "experimental"
    ),
    "three_network_shared_d_board": AlgorithmCapabilities(
        "three_network_shared_d_board", False, True, True, "experimental"
    ),
    "three_network_shared_d_paired": AlgorithmCapabilities(
        "three_network_shared_d_paired", False, True, True, "experimental"
    ),
    "three_network_shared_d_conditioned_paired": AlgorithmCapabilities(
        "three_network_shared_d_conditioned_paired", False, True, True, "experimental"
    ),
    "three_network_shared_d_conditioned": AlgorithmCapabilities(
        "three_network_shared_d_conditioned", False, True, True, "experimental"
    ),
    "three_network_shared_d_synchronous": AlgorithmCapabilities(
        "three_network_shared_d_synchronous", False, True, True, "experimental"
    ),
    "three_network_shared_d_two_timescale": AlgorithmCapabilities(
        "three_network_shared_d_two_timescale", False, True, True, "experimental"
    ),
    "td3": AlgorithmCapabilities("td3", False, True, True, "planned"),
    "sac": AlgorithmCapabilities("sac", False, True, True, "planned"),
    "a2c": AlgorithmCapabilities("a2c", True, False, True, "planned"),
    "a3c": AlgorithmCapabilities("a3c", True, False, True, "planned"),
}


def get_algorithm_metadata(name: str) -> AlgorithmCapabilities:
    """Return capabilities, or fail clearly for an unknown algorithm."""
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm: {name}") from exc


def create_algorithm(name: str, **kwargs: Any) -> Any:
    """Construct an implemented algorithm and refuse placeholders."""
    metadata = get_algorithm_metadata(name)
    if metadata.implementation_status == "planned":
        raise NotImplementedError(f"{name} is planned; no training implementation is available")
    if name == "ddpg":
        from .ddpg import DDPG

        return DDPG.from_config(**kwargs)
    if name == "ddpg_no_target_actor":
        from .ddpg import DDPGNoTargetActor

        return DDPGNoTargetActor.from_config(**kwargs)
    if name == "three_network_ddpg":
        from .three_network_ddpg import ThreeNetworkDDPG

        return ThreeNetworkDDPG.from_config(**kwargs)
    if name == "three_network_shared_d":
        from .shared_d import SharedDDDPG

        return SharedDDDPG.from_config(**kwargs)
    if name == "three_network_shared_d_board":
        from .shared_d import BoardSharedDDDPG

        return BoardSharedDDDPG.from_config(**kwargs)
    if name == "three_network_shared_d_paired":
        from .paired_shared_d import PairedSharedDDDPG

        return PairedSharedDDDPG.from_config(**kwargs)
    if name == "three_network_shared_d_conditioned_paired":
        from .paired_shared_d import ConditionedPairedSharedDDDPG

        return ConditionedPairedSharedDDDPG.from_config(**kwargs)
    if name == "three_network_shared_d_conditioned":
        from .shared_d import ConditionedSharedDDDPG

        return ConditionedSharedDDDPG.from_config(**kwargs)
    if name == "three_network_shared_d_synchronous":
        from .shared_d import SynchronousSharedDDDPG

        return SynchronousSharedDDDPG.from_config(**kwargs)
    if name == "three_network_shared_d_two_timescale":
        from .shared_d import TwoTimescaleSharedDDDPG

        return TwoTimescaleSharedDDDPG.from_config(**kwargs)
    raise NotImplementedError(f"no constructor registered for {name}")
