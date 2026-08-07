import jax
import jax.numpy as jnp
import os
import optax
from functools import partial

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.formula import find_composition_vector
from crystalformer.src.lattice import norm_lattice


INITIAL_HEAD_MODULE_NAMES = frozenset({"linear", "linear_1", "linear_2", "linear_3"})


def _weighted_logp(logp_parts, lamb_g, lamb_w, lamb_xyz, lamb_a, lamb_l):
    logp_g, logp_w, logp_xyz, logp_a, logp_l = logp_parts
    return (
        lamb_g * logp_g
        + lamb_w * logp_w
        + lamb_xyz * logp_xyz
        + lamb_a * logp_a
        + lamb_l * logp_l
    )


def make_ppo_loss_fn(logp_fn, eps_clip, beta=0.1, alpha=0.0, gamma=1.0, lamb_g=1.0, lamb_xyz=1.0,lamb_a=1.0, lamb_w=1.0, lamb_l=1.0):

    """
    PPO clipped objective function with KL divergence regularization
    PPO_loss = PPO-clip + beta  * KL(P || P_pretrain)

    Note that we only consider the logp_xyz and logp_l in the logp_fn
    """

    def ppo_loss_fn(params, key, buffer, x, old_logp, pretrain_logp, advantages):

        logp_parts = logp_fn(params, key, *x, False)
        logp_g, logp_w, logp_xyz, logp_a, logp_l = logp_parts
        logp = _weighted_logp(
            logp_parts, lamb_g, lamb_w, lamb_xyz, lamb_a, lamb_l
        )

        # Finding the ratio (pi_theta / pi_theta__old)
        log_ratio = logp - old_logp
        ratios = jnp.exp(log_ratio)

        # Finding Surrogate Loss  
        surr1 = ratios * advantages
        surr2 = jax.lax.clamp(1-eps_clip, ratios, 1+eps_clip) * advantages

        # Final loss of clipped objective PPO
        reference_log_ratio = logp - pretrain_logp
        ppo_loss = jnp.mean(jnp.minimum(surr1, surr2))
        ppo_loss -= beta * jnp.mean(reference_log_ratio)
        ppo_loss -= alpha * jnp.mean(logp)
        if gamma > 0.0:
            logp_buffer = _weighted_logp(
                logp_fn(params, key, *buffer, False),
                lamb_g, lamb_w, lamb_xyz, lamb_a, lamb_l,
            )
            ppo_loss += gamma * jnp.mean(logp_buffer)

        clip_fraction = jnp.mean(jnp.abs(ratios - 1.0) > eps_clip)
        approx_kl_old = jnp.mean((ratios - 1.0) - log_ratio)
        diagnostics = (
            jnp.mean(reference_log_ratio),
            -jnp.mean(logp_g), -jnp.mean(logp_w), -jnp.mean(logp_a),
            -jnp.mean(logp_xyz), -jnp.mean(logp_l),
            clip_fraction, jnp.mean(ratios), jnp.max(ratios), approx_kl_old,
        )
        return ppo_loss, diagnostics
    
    ppo_loss_fn.logp_weights = (lamb_g, lamb_w, lamb_xyz, lamb_a, lamb_l)
    return ppo_loss_fn


def update_replay_buffer(exp_buffer, G, L, XYZ, A, W, scores, batchsize, epoch=0):
    """Keep top raw, direction-adjusted scores and their source epochs."""
    # Concatenate current batch with buffer for each component
    G_combined = jnp.concatenate([exp_buffer[0], G], axis=0)
    L_combined = jnp.concatenate([exp_buffer[1], L], axis=0)
    XYZ_combined = jnp.concatenate([exp_buffer[2], XYZ], axis=0)
    A_combined = jnp.concatenate([exp_buffer[3], A], axis=0)
    W_combined = jnp.concatenate([exp_buffer[4], W], axis=0)
    scores_combined = jnp.concatenate([exp_buffer[5], scores], axis=0)
    old_epochs = (
        exp_buffer[6]
        if len(exp_buffer) > 6
        else jnp.full(exp_buffer[5].shape, -1, dtype=jnp.int32)
    )
    epochs_combined = jnp.concatenate([
        old_epochs,
        jnp.full(scores.shape, epoch, dtype=jnp.int32),
    ])

    # Keep top-k samples (descending sort by reward)
    # Keep at least one sample for low-memory smoke tests and tiny target runs.
    buffersize = max(1, batchsize // 10)
    # Stable sorting prefers the newly concatenated record only when its raw
    # score is genuinely better; a new record therefore cannot be lost because
    # its historical-best-shifted reward happened to tie at zero.
    topk_idx = jnp.argsort(-scores_combined, stable=True)[:buffersize]

    return (
        G_combined[topk_idx],
        L_combined[topk_idx],
        XYZ_combined[topk_idx],
        A_combined[topk_idx],
        W_combined[topk_idx],
        scores_combined[topk_idx],
        epochs_combined[topk_idx],
    )


def standardize_advantages(values, eps=1e-8):
    """Return finite batch-standardized advantages."""

    values = jnp.asarray(values)
    if values.size == 0:
        raise ValueError("values must contain at least one item")
    centered = values - jnp.mean(values)
    return centered / (jnp.std(values) + eps)


def output_head_module_names(tree):
    """Infer the shared final head without depending on Transformer depth."""

    modules = set(tree.keys()) if hasattr(tree, "keys") else set()
    numbered = [
        (int(name.removeprefix("linear_")), name)
        for name in modules
        if name.startswith("linear_") and name.removeprefix("linear_").isdigit()
    ]
    final_head = max(numbered)[1] if numbered else None
    return INITIAL_HEAD_MODULE_NAMES | ({final_head} if final_head else set())


def mask_tree_for_trainable_scope(tree, scope, head_modules=None):
    """Zero leaves outside the selected Haiku output-head modules."""

    if scope == "all":
        return tree
    if scope != "heads":
        raise ValueError("trainable scope must be 'all' or 'heads'")

    if head_modules is None:
        head_modules = output_head_module_names(tree)

    def mask(path, leaf):
        module = getattr(path[0], "key", None) if path else None
        return leaf if module in head_modules else jnp.zeros_like(leaf)

    return jax.tree_util.tree_map_with_path(mask, tree)


def metric_to_rewards(metric, direction="minimize", global_extreme=None):
    """Convert a raw metric to centered PPO rewards.

    The training loop historically minimizes energy-like metrics.  XRD is a
    similarity and therefore has the opposite ordering.  Keeping this small
    conversion in one function makes the convention explicit and testable.
    ``global_extreme`` is carried between epochs so the replay buffer keeps a
    consistent reference point.
    """

    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be 'minimize' or 'maximize'")
    metric = jnp.asarray(metric)
    if metric.size == 0:
        raise ValueError("metric must contain at least one value")
    if direction == "maximize":
        current = jnp.max(metric)
        extreme = current if global_extreme is None else jnp.maximum(global_extreme, current)
        rewards = metric - extreme
    else:
        current = jnp.min(metric)
        extreme = current if global_extreme is None else jnp.minimum(global_extreme, current)
        rewards = extreme - metric
    return rewards, extreme


def structural_diversity_bonus(G, W, A):
    """Return a centered inverse-frequency bonus for sampled token sequences.

    The bonus is intentionally based on the discrete structure decisions (space
    group, Wyckoff sequence, and atom sequence).  Coordinates and lattice
    values remain continuous and are already regularized by the PPO prior.
    """

    G = jnp.asarray(G).reshape(-1, 1)
    W = jnp.asarray(W)
    A = jnp.asarray(A)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if W.ndim != 2 or A.ndim != 2:
        raise ValueError("W and A must be one- or two-dimensional token arrays")
    if G.shape[0] == 0 or W.shape[0] != G.shape[0] or A.shape[0] != G.shape[0]:
        raise ValueError("G, W and A must describe the same non-empty batch")
    signatures = jnp.concatenate((G, W, A), axis=1)
    _, inverse, counts = jnp.unique(
        signatures, axis=0, return_inverse=True, return_counts=True
    )
    bonus = 1.0 / counts[inverse]
    return bonus - jnp.mean(bonus)


def microbatch_slices(batchsize, num_devices, microbatch_size):
    """Return per-device slices for equal-size global microbatches."""

    if batchsize <= 0 or num_devices <= 0 or microbatch_size <= 0:
        raise ValueError("batch sizes and device count must be positive")
    if batchsize % num_devices:
        raise ValueError("batchsize must be divisible by num_devices")
    if microbatch_size % num_devices:
        raise ValueError("microbatch_size must be divisible by num_devices")
    if batchsize % microbatch_size:
        raise ValueError("batchsize must be divisible by microbatch_size")
    per_device_batch = batchsize // num_devices
    per_device_microbatch = microbatch_size // num_devices
    return [
        (start, start + per_device_microbatch)
        for start in range(0, per_device_batch, per_device_microbatch)
    ]


def train(
    key,
    optimizer,
    opt_state,
    logp_fn,
    batch_reward_fn,
    ppo_loss_fn,
    sample_crystal,
    composition,
    params,
    epoch_finished,
    epochs,
    ppo_epochs,
    batchsize,
    path,
    reward_direction="minimize",
    sample_multiplier=10,
    sampling_batchsize=None,
    ppo_microbatch_size=None,
    max_sampling_attempts=1000,
    metric_name="e",
    diversity_weight=0.0,
    checkpoint_interval=10,
    standardize_raw_metric=False,
    replay_weight=1.0,
    trainable_scope="all",
):

    if reward_direction not in {"minimize", "maximize"}:
        raise ValueError("reward_direction must be 'minimize' or 'maximize'")
    if sample_multiplier <= 0:
        raise ValueError("sample_multiplier must be positive")
    if sampling_batchsize is not None and sampling_batchsize <= 0:
        raise ValueError("sampling_batchsize must be positive")
    if ppo_microbatch_size is not None and ppo_microbatch_size <= 0:
        raise ValueError("ppo_microbatch_size must be positive")
    if max_sampling_attempts <= 0:
        raise ValueError("max_sampling_attempts must be positive")
    if diversity_weight < 0:
        raise ValueError("diversity_weight must be non-negative")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if replay_weight < 0:
        raise ValueError("replay_weight must be non-negative")
    if trainable_scope not in {"all", "heads"}:
        raise ValueError("trainable_scope must be 'all' or 'heads'")

    is_comp_provided = jnp.sum(composition) > 0
    trainable_head_modules = output_head_module_names(params)
    num_devices = jax.local_device_count()
    if batchsize < num_devices or batchsize % num_devices != 0:
        raise ValueError(
            f"batchsize ({batchsize}) must be a positive multiple of local devices ({num_devices})"
        )
    batch_per_device = batchsize // num_devices
    shape_prefix = (num_devices, batch_per_device)
    micro_slices = (
        None
        if ppo_microbatch_size is None
        else microbatch_slices(batchsize, num_devices, ppo_microbatch_size)
    )
    print("num_devices: ", num_devices)
    print("batchsize: ", batchsize)
    print("batch_per_device: ", batch_per_device)
    print("shape_prefix: ", shape_prefix)
    print("sampling_batchsize: ", sampling_batchsize)
    print("ppo_microbatch_size: ", ppo_microbatch_size)
    print("is_comp_provided: ", is_comp_provided)
    print("trainable_scope: ", trainable_scope)
    if trainable_scope == "heads":
        print("trainable_head_modules: ", sorted(trainable_head_modules))

    @partial(
        jax.pmap,
        axis_name="p",
        in_axes=(None, None, None, None, 0, 0, 0, 0),
        out_axes=(None, None, 0, None),
    )
    def step(params, key, opt_state, buffer, x, old_logp, pretrain_logp, advantages):
        value, grad = jax.value_and_grad(ppo_loss_fn, has_aux=True)(params, key, buffer, x, old_logp, pretrain_logp, advantages)
        grad = jax.lax.pmean(grad, axis_name="p")
        value = jax.lax.pmean(value, axis_name="p")
        grad = jax.tree_util.tree_map(lambda g_: g_ * -1.0, grad)  # invert gradient for maximization
        grad = mask_tree_for_trainable_scope(
            grad, trainable_scope, trainable_head_modules
        )
        grad_norm = optax.global_norm(grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        updates = mask_tree_for_trainable_scope(
            updates, trainable_scope, trainable_head_modules
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, grad_norm

    @partial(
        jax.pmap,
        axis_name="p",
        in_axes=(None, None, None, 0, 0, 0, 0),
        out_axes=(None, 0, None),
    )
    def grad_step(params, key, buffer, x, old_logp, pretrain_logp, advantages):
        value, grad = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            params, key, buffer, x, old_logp, pretrain_logp, advantages
        )
        grad = jax.lax.pmean(grad, axis_name="p")
        value = jax.lax.pmean(value, axis_name="p")
        grad = jax.tree_util.tree_map(lambda g_: g_ * -1.0, grad)
        grad = mask_tree_for_trainable_scope(
            grad, trainable_scope, trainable_head_modules
        )
        return grad, value, optax.global_norm(grad)

    @jax.jit
    def apply_grad(params, opt_state, grad):
        updates, opt_state = optimizer.update(grad, opt_state, params)
        updates = mask_tree_for_trainable_scope(
            updates, trainable_scope, trainable_head_modules
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state

    os.makedirs(path, exist_ok=True)
    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        if is_comp_provided:
            f.write(
                f"epoch {metric_name}_mean {metric_name}_err {metric_name}_max {metric_name}_min "
                "reward_mean reward_err reward_max reward_min advantage_mean advantage_std "
                "ppo_objective attempt unique_space_groups unique_wyckoff_sequences "
                "unique_atom_sequences unique_WA_combinations log_ratio_to_reference g w a xyz l "
                "clip_fraction ratio_mean ratio_max approx_kl_old grad_norm "
                "score_p10 score_p50 score_p90 replay_epoch replay_score\n"
            )
        else:
            f.write("epoch f_mean f_err f_max\n")

    pretrain_params = params
    logp_weights = getattr(ppo_loss_fn, "logp_weights", (1.0,) * 5)
    logp_fn = jax.jit(logp_fn, static_argnums=8)
    
    global_metric_extreme = jnp.inf if reward_direction == "minimize" else -jnp.inf
    for epoch in range(epoch_finished+1, epoch_finished+epochs+1):

        # Accumulate samples until we have batchsize that match the formula
        accumulated_G = []
        accumulated_XYZ = []
        accumulated_A = []
        accumulated_W = []
        accumulated_M = []
        accumulated_L = []
        
        num_matched = 0
        max_attempts = max_sampling_attempts
        attempt = 0
        
        while num_matched < batchsize and attempt < max_attempts:
            key, subkey = jax.random.split(key)

            if sampling_batchsize is not None:
                sample_bs = sampling_batchsize
            else:
                sample_bs = (
                    max(batchsize, int(sample_multiplier * batchsize))
                    if is_comp_provided
                    else batchsize
                )
            G, XYZ, A, W, M, L = sample_crystal(subkey, params, sample_bs, composition)

            if is_comp_provided:
                actual_compositions = jax.vmap(find_composition_vector)(A, M)
                formula_match = jnp.all(actual_compositions == composition, axis=1)
            else:
                formula_match = jnp.ones(G.shape[0], dtype=bool)
            
            # Get number of matched samples
            num_new_matched = int(jnp.sum(formula_match))
            
            if num_new_matched > 0:
                # Calculate how many we still need
                remaining = batchsize - num_matched
                to_take = min(num_new_matched, remaining)
                
                # Extract matched samples using boolean indexing
                G_matched = G[formula_match]
                XYZ_matched = XYZ[formula_match]
                A_matched = A[formula_match]
                W_matched = W[formula_match]
                M_matched = M[formula_match]
                L_matched = L[formula_match]
                
                # Take only the number we need
                accumulated_G.append(G_matched[:to_take])
                accumulated_XYZ.append(XYZ_matched[:to_take])
                accumulated_A.append(A_matched[:to_take])
                accumulated_W.append(W_matched[:to_take])
                accumulated_M.append(M_matched[:to_take])
                accumulated_L.append(L_matched[:to_take])
                
                num_matched += to_take
                print (f'collected {num_matched} samples with {attempt+1} attempts') 

            attempt += 1
        
        if num_matched < batchsize:
            raise RuntimeError(f"Epoch {epoch} - Could only generate {num_matched} formula-matching crystals after {max_attempts} attempts, but needed {batchsize}. Stopping training.")
        
        # Concatenate all accumulated samples
        G = jnp.concatenate(accumulated_G, axis=0)
        XYZ = jnp.concatenate(accumulated_XYZ, axis=0)
        A = jnp.concatenate(accumulated_A, axis=0)
        W = jnp.concatenate(accumulated_W, axis=0)
        M = jnp.concatenate(accumulated_M, axis=0)
        L = jnp.concatenate(accumulated_L, axis=0)
        
        # Truncate to exactly batchsize (should already be exact, but just in case)
        G = G[:batchsize]
        XYZ = XYZ[:batchsize]
        A = A[:batchsize]
        W = W[:batchsize]
        M = M[:batchsize]
        L = L[:batchsize]

        print (f'sampling done, move on to ppo') 
        
        if is_comp_provided:
            # Compute unique number of space groups 
            unique_space_groups = jnp.unique(G, return_counts=False).shape[0]
            # Number of unique wyckoff sequences for those matched formula
            unique_wyckoff_sequences = jnp.unique(W, axis=0, return_counts=False).shape[0]
            unique_atom_sequences = jnp.unique(A, axis=0, return_counts=False).shape[0]
            # Number of unique (W, A) combinations
            WA_combined = jnp.concatenate([W, A], axis=1)
            unique_WA_combinations = jnp.unique(WA_combined, axis=0, return_counts=False).shape[0]

        x = (G, L, XYZ, A, W)

        if is_comp_provided:
            # only using this type of reward when composition is provided
            metric = jnp.asarray(batch_reward_fn(x, path, epoch))

            metric_min = jnp.min(metric)
            metric_max = jnp.max(metric)
            metric_mean = jnp.mean(metric)
            metric_err = jnp.std(metric) / jnp.sqrt(batchsize)

            if standardize_raw_metric:
                rewards = metric if reward_direction == "maximize" else -metric
            else:
                rewards, global_metric_extreme = metric_to_rewards(
                    metric, reward_direction, global_metric_extreme
                )
        else:
            metric = jnp.asarray(batch_reward_fn(x))
            if reward_direction == "minimize":
                # Preserve the historical unconstrained/DNG convention.
                rewards = -metric
            else:
                rewards, global_metric_extreme = metric_to_rewards(
                    metric, reward_direction, global_metric_extreme
                )
            f_mean = jnp.mean(rewards)
            f_err = jnp.std(rewards) / jnp.sqrt(batchsize)

        replay_scores = metric if reward_direction == "maximize" else -metric
        if diversity_weight > 0.0:
            rewards = rewards + diversity_weight * structural_diversity_bonus(G, W, A)

        if standardize_raw_metric:
            advantages = standardize_advantages(rewards)
        else:
            baseline = rewards.mean() if epoch == epoch_finished+1 else 0.95 * baseline + 0.05 * rewards.mean()
            advantages = rewards - baseline
        reward_mean = jnp.mean(rewards)
        reward_err = jnp.std(rewards) / jnp.sqrt(batchsize)
        reward_max = jnp.max(rewards)
        reward_min = jnp.min(rewards)
        advantage_mean = jnp.mean(advantages)
        advantage_std = jnp.std(advantages)

        G, L, XYZ, A, W = x
        L = norm_lattice(G, W, L)
        x = (G, L, XYZ, A, W)
        # add composition information
        x = (composition[None, :].repeat(x[0].shape[0], axis=0),) + x
        
        if epoch == epoch_finished+1:
            exp_buffer = (
                jnp.empty_like(G[:0]),     # G
                jnp.empty_like(L[:0]),     # L
                jnp.empty_like(XYZ[:0]),   # XYZ
                jnp.empty_like(A[:0]),     # A
                jnp.empty_like(W[:0]),     # W
                jnp.empty((0,)),           # direction-adjusted raw scores
                jnp.empty((0,), dtype=jnp.int32),
            )

        if replay_weight > 0.0:
            exp_buffer = update_replay_buffer(
                exp_buffer, G, L, XYZ, A, W, replay_scores, batchsize, epoch
            )
        buffer = exp_buffer[:5]
        # add composition information
        buffer = (composition[None, :].repeat(buffer[0].shape[0], axis=0),) + buffer

        def evaluate_logp(evaluation_params, evaluation_key):
            if ppo_microbatch_size is None:
                return logp_fn(evaluation_params, evaluation_key, *x, False)
            chunks = []
            for start in range(0, batchsize, ppo_microbatch_size):
                evaluation_key, chunk_key = jax.random.split(evaluation_key)
                stop = start + ppo_microbatch_size
                chunk_x = jax.tree_util.tree_map(lambda item: item[start:stop], x)
                chunks.append(logp_fn(evaluation_params, chunk_key, *chunk_x, False))
            return tuple(
                jnp.concatenate([chunk[index] for chunk in chunks], axis=0)
                for index in range(5)
            )

        key, subkey1, subkey2 = jax.random.split(key, 3)
        logp_g, logp_w, logp_xyz, logp_a, logp_l = evaluate_logp(params, subkey1)
        old_logp = _weighted_logp(
            (logp_g, logp_w, logp_xyz, logp_a, logp_l),
            *logp_weights,
        )

        logp_g, logp_w, logp_xyz, logp_a, logp_l = evaluate_logp(pretrain_params, subkey2)
        pretrain_logp = _weighted_logp(
            (logp_g, logp_w, logp_xyz, logp_a, logp_l),
            *logp_weights,
        )

        x = jax.tree_util.tree_map(lambda _x: _x.reshape(shape_prefix + _x.shape[1:]), x)
        old_logp = old_logp.reshape(shape_prefix + old_logp.shape[1:])
        pretrain_logp = pretrain_logp.reshape(shape_prefix + pretrain_logp.shape[1:])
        advantages = advantages.reshape(shape_prefix + advantages.shape[1:])

        for _ in range(ppo_epochs):
            if micro_slices is None:
                key, subkey = jax.random.split(key)
                params, opt_state, value, grad_norm = step(
                    params, subkey, opt_state, buffer, x,
                    old_logp, pretrain_logp, advantages
                )
            else:
                accumulated_grad = None
                micro_values = []
                for start, stop in micro_slices:
                    key, subkey = jax.random.split(key)
                    micro_x = jax.tree_util.tree_map(
                        lambda item: item[:, start:stop], x
                    )
                    grad, micro_value, _ = grad_step(
                        params,
                        subkey,
                        buffer,
                        micro_x,
                        old_logp[:, start:stop],
                        pretrain_logp[:, start:stop],
                        advantages[:, start:stop],
                    )
                    accumulated_grad = (
                        grad
                        if accumulated_grad is None
                        else jax.tree_util.tree_map(
                            lambda total, item: total + item,
                            accumulated_grad,
                            grad,
                        )
                    )
                    micro_values.append(micro_value)
                accumulated_grad = jax.tree_util.tree_map(
                    lambda item: item / len(micro_slices), accumulated_grad
                )
                params, opt_state = apply_grad(params, opt_state, accumulated_grad)
                value = jax.tree_util.tree_map(
                    lambda *items: sum(items) / len(items), *micro_values
                )
                # Preserve the largest observed ratio instead of averaging the
                # per-microbatch maxima with the other scalar diagnostics.
                value = (
                    value[0],
                    value[1][:8]
                    + (jnp.max(jnp.stack([item[1][8] for item in micro_values])),)
                    + value[1][9:],
                )
                grad_norm = optax.global_norm(accumulated_grad)
            ppo_loss, diagnostics = value
            (
                reference_log_ratio, entropy_g, entropy_w, entropy_a,
                entropy_xyz, entropy_l, clip_fraction, ratio_mean, ratio_max,
                approx_kl_old,
            ) = diagnostics

        # PMAP retains a leading device axis for this scalar in the
        # microbatch path; flatten it before writing the host-side log.
        ppo_objective = jnp.ravel(ppo_loss)[0]
        
        if is_comp_provided:
            score_p10, score_p50, score_p90 = jnp.quantile(metric, jnp.array([0.1, 0.5, 0.9]))
            replay_epoch = int(exp_buffer[6][0]) if exp_buffer[6].size else -1
            replay_score = float(exp_buffer[5][0]) if exp_buffer[5].size else float("nan")
            scalar = lambda value: float(jnp.ravel(value)[0])
            f.write(
                ("%6d" + 11*"  %.6f" + 5*"  %3d" + 14*"  %.6f" + "  %6d  %.6f\n")
                % (
                    epoch, metric_mean, metric_err, metric_max, metric_min,
                    reward_mean, reward_err, reward_max, reward_min,
                    advantage_mean, advantage_std, ppo_objective,
                    attempt, unique_space_groups, unique_wyckoff_sequences,
                    unique_atom_sequences, unique_WA_combinations,
                    scalar(reference_log_ratio), scalar(entropy_g), scalar(entropy_w),
                    scalar(entropy_a), scalar(entropy_xyz), scalar(entropy_l),
                    scalar(clip_fraction), scalar(ratio_mean), scalar(ratio_max),
                    scalar(approx_kl_old), scalar(grad_norm),
                    score_p10, score_p50, score_p90, replay_epoch, replay_score,
                )
            )
        else:
            f.write( ("%6d" + 3*"  %.6f" +"\n") % (epoch, f_mean, f_err, jnp.max(rewards)) )

        if epoch % checkpoint_interval == 0 or epoch == epoch_finished + epochs:
            # PPO evaluation only needs model parameters. Omitting Adam moments
            # keeps checkpoint writes below the Windows/WSL commit limit.
            ckpt = {"params": params}
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()

    return params, opt_state
