import numpy as np
import jax
import jax.numpy as jnp
import pytest

from crystalformer.src.loss import make_loss_fn
from crystalformer.src.sample import composition_progress_mask
from crystalformer.src.composition_reachability import (
    atom_reachability_costs,
    atom_reachability_mask,
    fixed_multiplicity_assignment_reachable,
    spacegroup_reachability_mask,
    trajectory_reachability_masks,
    wyckoff_reachability_costs,
    wyckoff_reachability_mask,
)


def _composition(**counts):
    atomic_numbers = {"O": 8, "Mg": 12, "Cl": 17, "K": 19, "Fe": 26}
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


def test_fixed_multiplicity_counterexamples_are_reachable():
    assert fixed_multiplicity_assignment_reachable((4, 4, 8), (1, 1))
    assert fixed_multiplicity_assignment_reachable((4, 4, 4, 12), (1, 1))
    assert fixed_multiplicity_assignment_reachable((4, 8, 12), (1, 1))


def test_reachability_keeps_both_kcl_choices_after_first_fourfold_site():
    composition = _composition(K=1, Cl=1)
    atoms = jnp.asarray([19] + [0] * 20)
    wyckoff = jnp.asarray([1] + [0] * 20)

    mask = atom_reachability_mask(
        composition,
        atoms,
        wyckoff,
        spacegroup=225,
        current_w=2,
        remaining_after_current=19,
    )

    assert bool(mask[17])
    assert bool(mask[19])
    assert not bool(mask[8])
    assert not bool(mask[0])


def test_size_cost_prefers_smaller_reachable_wyckoff_path():
    composition = _composition(Mg=1, O=1)
    atoms = jnp.zeros(21, dtype=jnp.int32)
    wyckoff = jnp.zeros(21, dtype=jnp.int32)

    costs = wyckoff_reachability_costs(
        composition, atoms, wyckoff, spacegroup=225,
        remaining_after_current=19, max_atoms=128,
    )

    finite = costs[np.isfinite(costs)]
    assert finite.size > 1
    assert np.min(finite) < np.max(finite)


def test_size_cost_keeps_larger_atom_actions_reachable():
    composition = _composition(Mg=1, O=1)
    atoms = jnp.asarray([12] + [0] * 20)
    wyckoff = jnp.asarray([1] + [0] * 20)

    mask = atom_reachability_mask(
        composition, atoms, wyckoff, 225, current_w=2,
        remaining_after_current=19, max_atoms=128,
    )
    costs = atom_reachability_costs(
        composition, atoms, wyckoff, 225, current_w=2,
        remaining_after_current=19, max_atoms=128,
    )

    assert bool(mask[8]) and bool(mask[12])
    assert np.isfinite(costs[8]) and np.isfinite(costs[12])


def test_negative_size_bias_is_rejected():
    with pytest.raises(ValueError, match="composition_size_bias"):
        make_loss_fn(4, 119, 28, 2, 1, None, composition_size_bias=-0.1)


def test_reachability_only_allows_pad_after_ratio_is_complete():
    composition = _composition(K=1, Cl=1)
    incomplete_atoms = jnp.asarray([19] + [0] * 20)
    complete_atoms = jnp.asarray([19, 17] + [0] * 19)
    incomplete_wyckoff = jnp.asarray([1] + [0] * 20)
    complete_wyckoff = jnp.asarray([1, 2] + [0] * 19)

    incomplete = wyckoff_reachability_mask(
        composition, incomplete_atoms, incomplete_wyckoff, 225, 19
    )
    complete = wyckoff_reachability_mask(
        composition, complete_atoms, complete_wyckoff, 225, 18
    )

    assert not bool(incomplete[0])
    assert bool(complete[0])


def test_unfinishable_final_site_is_masked():
    composition = _composition(Fe=2, O=3)
    atoms = jnp.zeros(21, dtype=jnp.int32)
    wyckoff = jnp.zeros(21, dtype=jnp.int32)

    mask = wyckoff_reachability_mask(
        composition, atoms, wyckoff, spacegroup=225,
        remaining_after_current=0,
    )

    assert not bool(mask[0])
    assert not bool(mask[1])
    assert not bool(mask[2])
    assert not bool(mask[3])


def test_spacegroup_mask_rejects_formula_above_atom_limit():
    composition = _composition(Fe=2, O=3)

    impossible = spacegroup_reachability_mask(
        composition, max_sites=20, max_atoms=4
    )
    reachable = spacegroup_reachability_mask(
        _composition(K=1, Cl=1), max_sites=20, max_atoms=8
    )

    assert not np.any(impossible)
    assert bool(reachable[224])  # Space group 225 has two fourfold sites.


def test_sampler_rejects_globally_unreachable_formula():
    n_max = 3
    atom_types = 119
    wyck_types = 28
    Kx = 2
    Kl = 1
    output_size = atom_types + Kl + 12 * Kl

    def transformer(params, key, composition, group, xyz, atoms, wyckoff,
                    multiplicities, is_train):
        del params, key, composition, group, atoms, wyckoff, multiplicities
        del is_train
        return jax.nn.log_softmax(jnp.zeros(230)), jnp.zeros(
            (5 * len(xyz) + 1, output_size)
        )

    from crystalformer.src.sample import make_sample_crystal

    sampler = make_sample_crystal(
        transformer,
        n_max,
        atom_types,
        wyck_types,
        Kx,
        Kl,
        None,
        1.0,
        1.0,
        composition_max_atoms=4,
    )

    with pytest.raises(Exception, match="composition is unreachable"):
        sampler(
            jax.random.PRNGKey(0),
            None,
            1,
            _composition(Fe=2, O=3),
        )


def test_trajectory_reserves_final_pad_slot():
    composition = _composition(K=1, Cl=1)
    atoms = np.asarray([19, 17, 0], dtype=np.int32)
    wyckoff = np.asarray([1, 2, 0], dtype=np.int32)

    _, wyckoff_masks, atom_masks = trajectory_reachability_masks(
        composition, atoms, wyckoff, spacegroup=225
    )

    assert bool(wyckoff_masks[-1, 0])
    assert not np.any(wyckoff_masks[-1, 1:])
    assert bool(atom_masks[-1, 0])
    assert not np.any(atom_masks[-1, 1:])


def test_valid_kcl_trajectory_actions_remain_in_logp_support():
    composition = _composition(K=1, Cl=1)
    atoms = np.asarray([19, 19, 17, 0], dtype=np.int32)
    wyckoff = np.asarray([1, 2, 3, 0], dtype=np.int32)

    group_mask, wyckoff_masks, atom_masks = trajectory_reachability_masks(
        composition, atoms, wyckoff, spacegroup=225
    )

    assert bool(group_mask[224])
    assert all(bool(wyckoff_masks[i, w]) for i, w in enumerate(wyckoff))
    assert all(bool(atom_masks[i, atom]) for i, atom in enumerate(atoms))


def test_invalid_kcl_trajectory_is_removed_from_logp_support():
    composition = _composition(K=1, Cl=1)
    atoms = np.asarray([19, 17, 17, 0], dtype=np.int32)
    wyckoff = np.asarray([1, 2, 3, 0], dtype=np.int32)

    _, wyckoff_masks, atom_masks = trajectory_reachability_masks(
        composition, atoms, wyckoff, spacegroup=225
    )

    selected_support = [
        bool(wyckoff_masks[i, w] and atom_masks[i, atom])
        for i, (w, atom) in enumerate(zip(wyckoff, atoms))
    ]
    assert not all(selected_support)


def test_reachability_logp_support_is_jittable_and_differentiable():
    n_max = 4
    atom_types = 119
    wyck_types = 28
    Kx = 2
    Kl = 1
    output_size = atom_types + Kl + 12 * Kl

    def transformer(params, key, composition, group, xyz, atoms, wyckoff,
                    multiplicities, is_train):
        del key, composition, group, atoms, wyckoff, multiplicities, is_train
        group_logp = jax.nn.log_softmax(
            jnp.zeros(230).at[224].set(params)
        )
        outputs = jnp.zeros((5 * len(xyz) + 1, output_size))
        for offset in (2, 3, 4):
            outputs = outputs.at[offset::5, 2 * Kx:3 * Kx].set(1.0)
        sigma_start = atom_types + Kl + 6 * Kl
        outputs = outputs.at[1::5, sigma_start:sigma_start + 6 * Kl].set(1.0)
        return group_logp, outputs

    _, logp_fn = make_loss_fn(
        n_max,
        atom_types,
        wyck_types,
        Kx,
        Kl,
        transformer,
        composition_reachability=True,
        composition_size_bias=0.5,
    )
    composition = _composition(K=1, Cl=1)[None, :]
    groups = jnp.asarray([225])
    lattice = jnp.ones((1, 6))
    xyz = jnp.zeros((1, n_max, 3))
    atoms = jnp.asarray([[19, 19, 17, 0]])
    wyckoff = jnp.asarray([[1, 2, 3, 0]])
    key = jax.random.PRNGKey(0)

    def total_logp(params):
        values = logp_fn(
            params, key, composition, groups, lattice, xyz, atoms, wyckoff, False
        )
        return sum(jnp.sum(value) for value in values)

    value, gradient = jax.jit(jax.value_and_grad(total_logp))(jnp.asarray(0.1))

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))

    pmapped = jax.pmap(
        lambda params: jax.value_and_grad(total_logp)(params)
    )(jnp.asarray([0.1]))
    assert bool(jnp.all(jnp.isfinite(pmapped[0])))
    assert bool(jnp.all(jnp.isfinite(pmapped[1])))
