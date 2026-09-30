"""Verify the canonical research runtime without starting an experiment."""

import json
import platform

import gymnasium as gym
import numpy as np
import torch


def main() -> None:
    """Exercise PyTorch and one seeded Gymnasium transition on CPU."""
    tensor = torch.tensor([1.0, 2.0], device="cpu")
    if float(tensor.sum()) != 3.0:
        raise RuntimeError("PyTorch CPU tensor smoke check failed")

    environment = gym.make("Pendulum-v1")
    try:
        environment.action_space.seed(0)
        observation, _ = environment.reset(seed=0)
        action = environment.action_space.sample()
        next_observation, reward, terminated, truncated, _ = environment.step(action)
    finally:
        environment.close()

    if observation.shape != (3,) or next_observation.shape != (3,):
        raise RuntimeError("Pendulum-v1 returned an unexpected observation shape")
    if not np.isfinite(reward):
        raise RuntimeError("Pendulum-v1 returned a non-finite reward")

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gymnasium": gym.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "torch_cpu_sum": float(tensor.sum()),
                "pendulum_terminated": bool(terminated),
                "pendulum_truncated": bool(truncated),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
