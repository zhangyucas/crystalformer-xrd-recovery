import numpy as np
import pytest
import jax.numpy as jnp

from crystalformer.reinforce.ppo import (
    microbatch_slices,
    metric_to_rewards,
    structural_diversity_bonus,
    update_replay_buffer,
)


def _empty_buffer():
    return tuple(jnp.empty((0, 1), dtype=jnp.float32) for _ in range(5)) + (
        jnp.empty((0,), dtype=jnp.float32),
    )


def test_metric_to_rewards_maximize_orders_high_similarity_first():
    rewards, extreme = metric_to_rewards(jnp.array([0.2, 0.8, 0.5]), "maximize")
    np.testing.assert_allclose(np.asarray(rewards), [-0.6, 0.0, -0.3])
    assert float(extreme) == pytest.approx(0.8)


def test_metric_to_rewards_minimize_preserves_legacy_energy_order():
    rewards, extreme = metric_to_rewards(jnp.array([1.0, 3.0, 2.0]), "minimize")
    np.testing.assert_allclose(np.asarray(rewards), [0.0, -2.0, -1.0])
    assert float(extreme) == pytest.approx(1.0)


def test_metric_to_rewards_carries_global_extreme():
    rewards, extreme = metric_to_rewards(jnp.array([0.4, 0.6]), "maximize", 0.8)
    np.testing.assert_allclose(np.asarray(rewards), [-0.4, -0.2])
    assert float(extreme) == pytest.approx(0.8)


def test_update_replay_buffer_keeps_one_sample_for_tiny_batch():
    values = [jnp.array([[1.0], [2.0]]), jnp.array([[3.0], [4.0]])]
    buffer = _empty_buffer()
    updated = update_replay_buffer(
        buffer,
        values[0],
        values[1],
        values[0],
        values[0].astype(jnp.int32),
        values[0].astype(jnp.int32),
        jnp.array([-0.7, 0.0]),
        batchsize=4,
    )
    assert updated[0].shape[0] == 1
    assert float(updated[-1][0]) == pytest.approx(0.0)


def test_metric_to_rewards_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        metric_to_rewards(jnp.array([1.0]), "other")
    with pytest.raises(ValueError):
        metric_to_rewards(jnp.empty((0,)), "maximize")


def test_structural_diversity_bonus_prefers_rare_sequences():
    G = jnp.array([1, 1, 2])
    W = jnp.array([[1, 0], [1, 0], [1, 0]])
    A = jnp.array([[14, 0], [14, 0], [14, 0]])
    bonus = np.asarray(structural_diversity_bonus(G, W, A))
    assert bonus[2] > bonus[0]
    np.testing.assert_allclose(bonus.mean(), 0.0, atol=1e-6)


def test_microbatch_slices_preserve_device_partition():
    assert microbatch_slices(8, 1, 2) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert microbatch_slices(8, 2, 4) == [(0, 2), (2, 4)]
    with pytest.raises(ValueError):
        microbatch_slices(8, 2, 3)
