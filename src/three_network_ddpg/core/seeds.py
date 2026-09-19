"""Best-effort seeding with explicit limits on determinism."""

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedReport:
    seed: int
    numpy_seeded: bool
    torch_seeded: bool
    deterministic_algorithms_requested: bool


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> SeedReport:
    """Seed available libraries; exact reproducibility remains device-dependent."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        numpy_seeded = True

    torch_seeded = False
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        torch_seeded = True

    return SeedReport(seed, numpy_seeded, torch_seeded, deterministic_algorithms)


def seed_environment(environment: Any, seed: int) -> tuple[Any, Any]:
    """Seed reset and action sampling; return Gymnasium's reset result."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    environment.action_space.seed(seed)
    return environment.reset(seed=seed)
