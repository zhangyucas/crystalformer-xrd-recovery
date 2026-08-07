import numpy as np
import pytest
import jax
import jax.numpy as jnp

from crystalformer.reinforce.ppo import (
    make_ppo_loss_fn,
    mask_tree_for_trainable_scope,
    microbatch_slices,
    metric_to_rewards,
    standardize_advantages,
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


def test_standardized_advantages_have_zero_mean_and_unit_scale():
    advantages = np.asarray(standardize_advantages(jnp.array([0.2, 0.5, 0.9, 1.4])))
    assert advantages.mean() == pytest.approx(0.0, abs=1e-6)
    assert advantages.std() == pytest.approx(1.0, abs=1e-6)


def test_replay_uses_raw_score_so_new_records_replace_stale_sample():
    buffer = _empty_buffer()
    template = jnp.array([[1.0]])
    for epoch, score in enumerate((0.8, 0.9, 0.91), start=1):
        value = template * epoch
        buffer = update_replay_buffer(
            buffer, value, value, value, value.astype(int), value.astype(int),
            jnp.array([score]), batchsize=8, epoch=epoch,
        )
    assert float(buffer[5][0]) == pytest.approx(0.91)
    assert int(buffer[6][0]) == 3


def test_gamma_zero_skips_empty_replay_and_is_finite():
    def logp_fn(params, key, *args):
        del key, args
        value = jnp.asarray(params["logp"])
        return (value, jnp.zeros_like(value), jnp.zeros_like(value),
                jnp.zeros_like(value), jnp.zeros_like(value))

    loss = make_ppo_loss_fn(logp_fn, eps_clip=0.2, beta=0.0, gamma=0.0)
    value, diagnostics = loss(
        {"logp": jnp.array([0.1, -0.1])}, None, (), (),
        jnp.zeros(2), jnp.zeros(2), jnp.array([1.0, -1.0]),
    )
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(item)).all() for item in diagnostics)


def test_clipping_diagnostic_activates_after_policy_moves():
    def logp_fn(params, key, *args):
        del key, args
        value = jnp.asarray(params["logp"])
        zeros = jnp.zeros_like(value)
        return value, zeros, zeros, zeros, zeros

    loss = make_ppo_loss_fn(logp_fn, eps_clip=0.2, beta=0.0, gamma=0.0)
    _, first = loss({"logp": jnp.zeros(2)}, None, (), (), jnp.zeros(2), jnp.zeros(2), jnp.ones(2))
    _, later = loss({"logp": jnp.array([0.3, -0.3])}, None, (), (), jnp.zeros(2), jnp.zeros(2), jnp.ones(2))
    assert float(first[6]) == pytest.approx(0.0)
    assert float(later[6]) > 0.0


def test_heads_scope_masks_body_but_all_scope_preserves_tree():
    tree = {
        "linear": {"w": jnp.ones((2,))},
        "linear_41": {"w": jnp.ones((2,)) * 2},
        "linear_50": {"w": jnp.ones((2,)) * 4},
        "multi_head_attention/query": {"w": jnp.ones((2,)) * 3},
    }
    masked = mask_tree_for_trainable_scope(tree, "heads")
    np.testing.assert_array_equal(masked["linear"]["w"], tree["linear"]["w"])
    np.testing.assert_array_equal(masked["linear_41"]["w"], 0.0)
    np.testing.assert_array_equal(masked["linear_50"]["w"], tree["linear_50"]["w"])
    np.testing.assert_array_equal(masked["multi_head_attention/query"]["w"], 0.0)
    assert mask_tree_for_trainable_scope(tree, "all") is tree


def test_head_only_update_changes_heads_and_keeps_body_byte_identical():
    import optax

    params = {
        "linear": {"w": jnp.array([1.0, 2.0])},
        "multi_head_attention/query": {"w": jnp.array([3.0, 4.0])},
    }
    optimizer = optax.sgd(0.1)
    gradients = mask_tree_for_trainable_scope(
        jax.tree_util.tree_map(jnp.ones_like, params), "heads"
    )
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updates = mask_tree_for_trainable_scope(updates, "heads")
    changed = optax.apply_updates(params, updates)

    assert not np.array_equal(changed["linear"]["w"], params["linear"]["w"])
    assert changed["multi_head_attention/query"]["w"].tobytes() == (
        params["multi_head_attention/query"]["w"].tobytes()
    )


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
