"""Metadata-only registry. No entry currently has a training factory."""

from ..core.interfaces import AlgorithmCapabilities

ALGORITHMS: dict[str, AlgorithmCapabilities] = {
    "ddpg": AlgorithmCapabilities("ddpg", False, True, True, "planned"),
    "three_network_ddpg": AlgorithmCapabilities("three_network_ddpg", False, True, True, "planned"),
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


def create_algorithm(name: str) -> None:
    """Refuse construction until a reviewed implementation is registered."""
    get_algorithm_metadata(name)
    raise NotImplementedError(f"{name} is planned; no training implementation is available")
