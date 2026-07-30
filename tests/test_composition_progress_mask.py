import jax.numpy as jnp

from crystalformer.src.sample import composition_progress_mask


def _composition(**counts):
    atomic_numbers = {"O": 8, "Mg": 12, "Fe": 26}
    result = jnp.zeros(119, dtype=jnp.int32)
    for element, count in counts.items():
        result = result.at[atomic_numbers[element]].set(count)
    return result


def test_progress_mask_prioritizes_lagging_element_and_masks_non_targets():
    mask = composition_progress_mask(
        _composition(Mg=1, O=1),
        jnp.asarray([[12, 0]]),
        jnp.asarray([[4, 0]]),
    )[0]

    assert bool(mask[0])
    assert bool(mask[8])
    assert not bool(mask[12])
    assert not bool(mask[14])


def test_progress_mask_reenables_all_targets_when_progress_is_equal():
    mask = composition_progress_mask(
        _composition(Mg=1, O=1),
        jnp.asarray([[12, 8]]),
        jnp.asarray([[4, 4]]),
    )[0]

    assert bool(mask[8])
    assert bool(mask[12])


def test_progress_uses_reduced_ratio_and_strict_tolerance_boundary():
    composition = _composition(Fe=4, O=6)  # Reduced target ratio is Fe2O3.
    atoms = jnp.asarray([[26, 8]])
    multiplicities = jnp.asarray([[4, 3]])

    strict = composition_progress_mask(composition, atoms, multiplicities, 0.0)[0]
    tolerant = composition_progress_mask(composition, atoms, multiplicities, 1.0)[0]

    assert bool(strict[8])
    assert not bool(strict[26])
    assert bool(tolerant[8])
    assert bool(tolerant[26])
