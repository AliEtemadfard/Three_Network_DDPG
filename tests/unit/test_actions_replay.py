"""Action transforms, replay shapes, and exact bootstrap masking."""

import numpy as np
import pytest
import torch

from three_network_ddpg.core.actions import clip_action, scale_action, unscale_action
from three_network_ddpg.core.batches import Transition
from three_network_ddpg.core.replay import ReplayBuffer, bootstrap_mask


def test_action_scaling_inverse_and_explicit_clip() -> None:
    low = np.array([-2.0, 0.5], dtype=np.float32)
    high = np.array([3.0, 4.0], dtype=np.float32)
    normalized = np.array([-0.75, 0.25], dtype=np.float32)
    scaled = scale_action(normalized, low, high)
    np.testing.assert_allclose(unscale_action(scaled, low, high), normalized, atol=1e-6)
    np.testing.assert_array_equal(clip_action(np.array([-3.0, 9.0]), low, high), np.array([-2.0, 4.0]))
    with pytest.raises(ValueError, match="shapes"):
        scale_action(np.array([0.0]), low, high)


def _transition(index: int, *, terminated: bool = False, truncated: bool = False) -> Transition:
    return Transition(
        observation=np.array([index, index + 1], dtype=np.float32),
        action=np.array([0.5], dtype=np.float32),
        reward=float(index),
        next_observation=np.array([index + 1, index + 2], dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
    )


@pytest.mark.parametrize("bootstrap_on_truncation", [False, True])
def test_replay_dtypes_shapes_and_mask(bootstrap_on_truncation: bool) -> None:
    replay = ReplayBuffer(
        capacity=3,
        observation_shape=(2,),
        action_shape=(1,),
        seed=9,
        bootstrap_on_truncation=bootstrap_on_truncation,
        device="cpu",
    )
    replay.add(_transition(0, terminated=True))
    replay.add(_transition(1, truncated=True))
    replay.add(_transition(2))
    batch = replay.sample(3)
    assert batch.observations.shape == (3, 2)
    assert batch.actions.shape == (3, 1)
    assert batch.rewards.shape == (3, 1)
    assert batch.terminated.shape == (3, 1)
    assert batch.truncated.shape == (3, 1)
    assert batch.bootstrap_mask.shape == (3, 1)
    assert batch.observations.dtype == torch.float32
    assert batch.rewards.dtype == torch.float32
    assert batch.terminated.dtype == torch.bool
    assert batch.truncated.dtype == torch.bool
    assert batch.device == torch.device("cpu")
    assert batch.to("cpu").device == torch.device("cpu")
    masks = {
        int(reward): float(mask)
        for reward, mask in zip(batch.rewards[:, 0], batch.bootstrap_mask[:, 0], strict=True)
    }
    assert masks == {0: 0.0, 1: float(bootstrap_on_truncation), 2: 1.0}
    assert int(batch.terminated.sum()) == 1
    assert int(batch.truncated.sum()) == 1
    batch.observations.fill_(999.0)
    assert not torch.any(replay.sample(3).observations == 999.0)


def test_replay_ring_and_invalid_shapes() -> None:
    replay = ReplayBuffer(
        capacity=2,
        observation_shape=(2,),
        action_shape=(1,),
        seed=3,
        bootstrap_on_truncation=True,
        device="cpu",
    )
    for index in range(3):
        replay.add(_transition(index))
    assert len(replay) == 2
    assert set(replay.sample(2).rewards[:, 0].tolist()) == {1.0, 2.0}
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(3)
    with pytest.raises(ValueError, match="observation shape"):
        replay.add(
            Transition(
                observation=np.zeros(3),
                action=np.zeros(1),
                reward=0.0,
                next_observation=np.zeros(3),
                terminated=False,
                truncated=False,
            )
        )


def test_mask_keeps_termination_and_truncation_distinct() -> None:
    terminated = torch.tensor([[True], [False], [False], [True]])
    truncated = torch.tensor([[False], [True], [False], [True]])
    assert bootstrap_mask(terminated, truncated, bootstrap_on_truncation=True).flatten().tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    assert bootstrap_mask(terminated, truncated, bootstrap_on_truncation=False).flatten().tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]
