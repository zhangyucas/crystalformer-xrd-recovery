"""Exact stoichiometric reachability checks for ordered Wyckoff sampling."""

from functools import lru_cache
from math import ceil

import numpy as np

from crystalformer.src.wyckoff import dof0_table, mult_table, wmax_table


_MULTIPLICITIES = np.asarray(mult_table, dtype=np.int32)
_DOF_ZERO = np.asarray(dof0_table, dtype=bool)
_WMAX = np.asarray(wmax_table, dtype=np.int32)


def reduced_ratio(composition):
    composition = np.asarray(composition, dtype=np.int32)
    target = composition[composition > 0]
    if target.size == 0:
        return (), ()
    divisor = int(np.gcd.reduce(target))
    elements = tuple(int(value) for value in np.flatnonzero(composition > 0))
    ratio = tuple(int(composition[element] // divisor) for element in elements)
    return elements, ratio


def _is_complete(counts, ratio):
    if not counts or any(count <= 0 for count in counts):
        return False
    multiplier = counts[0] // ratio[0]
    return multiplier > 0 and all(
        count == multiplier * part for count, part in zip(counts, ratio)
    )


def _next_wyckoff_indices(spacegroup, previous_w):
    maximum = int(_WMAX[spacegroup - 1])
    if previous_w <= 0:
        return range(1, maximum + 1)
    start = previous_w + int(_DOF_ZERO[spacegroup - 1, previous_w])
    return range(start, maximum + 1)


@lru_cache(maxsize=500_000)
def _can_fill(spacegroup, previous_w, deficits, remaining_sites):
    """Fill one fixed target exactly; deficits provide strong DP pruning."""
    if not any(deficits):
        return True
    if remaining_sites <= 0:
        return False

    next_indices = tuple(_next_wyckoff_indices(spacegroup, previous_w))
    if not next_indices:
        return False
    positive_multiplicities = tuple(
        int(_MULTIPLICITIES[spacegroup - 1, w]) for w in next_indices
    )
    minimum_multiplicity = min(positive_multiplicities)
    maximum_multiplicity = max(positive_multiplicities)
    deficit_total = sum(deficits)
    if deficit_total < minimum_multiplicity:
        return False
    if deficit_total > remaining_sites * maximum_multiplicity:
        return False
    if not any(deficit >= minimum_multiplicity for deficit in deficits):
        return False

    element_order = sorted(range(len(deficits)), key=deficits.__getitem__, reverse=True)
    for w, multiplicity in zip(next_indices, positive_multiplicities):
        for element_index in element_order:
            if deficits[element_index] < multiplicity:
                continue
            updated = list(deficits)
            updated[element_index] -= multiplicity
            if _can_fill(
                spacegroup, w, tuple(updated), remaining_sites - 1
            ):
                return True
    return False


@lru_cache(maxsize=500_000)
def _can_finish(spacegroup, ratio, previous_w, counts, remaining_sites, max_atoms):
    if _is_complete(counts, ratio):
        return True
    if remaining_sites <= 0 or sum(counts) >= max_atoms:
        return False

    minimum_multiplier = max(
        1, max(ceil(count / part) for count, part in zip(counts, ratio))
    )
    maximum_multiplier = max_atoms // sum(ratio)
    for multiplier in range(minimum_multiplier, maximum_multiplier + 1):
        target = tuple(multiplier * part for part in ratio)
        if any(count > goal for count, goal in zip(counts, target)):
            continue
        deficits = tuple(goal - count for count, goal in zip(counts, target))
        deficit_total = sum(deficits)
        next_indices = tuple(_next_wyckoff_indices(spacegroup, previous_w))
        if not next_indices:
            continue
        multiplicities = tuple(
            int(_MULTIPLICITIES[spacegroup - 1, w]) for w in next_indices
        )
        if deficit_total < min(multiplicities):
            continue
        if deficit_total > remaining_sites * max(multiplicities):
            continue
        common_divisor = int(np.gcd.reduce(multiplicities))
        if any(deficit % common_divisor for deficit in deficits):
            continue
        if _can_fill(spacegroup, previous_w, deficits, remaining_sites):
            return True
    return False


def can_finish_composition(
    spacegroup,
    ratio,
    previous_w,
    counts,
    remaining_sites,
    max_atoms=512,
):
    """Return whether an ordered future Wyckoff sequence can reach ``k * ratio``."""
    return _can_finish(
        int(spacegroup),
        tuple(int(value) for value in ratio),
        int(previous_w),
        tuple(int(value) for value in counts),
        int(remaining_sites),
        int(max_atoms),
    )


def spacegroup_reachability_mask(composition, max_sites=21, max_atoms=512):
    """Return the space groups that can realize the reduced composition."""
    _, ratio = reduced_ratio(composition)
    if not ratio:
        return np.ones(230, dtype=bool)
    empty = (0,) * len(ratio)
    return np.asarray(
        [
            _can_finish(group, ratio, 0, empty, int(max_sites), int(max_atoms))
            for group in range(1, 231)
        ],
        dtype=bool,
    )


def fixed_multiplicity_assignment_reachable(multiplicities, ratio):
    """Check whether a fixed multiplicity sequence can be split into ``k * ratio``."""
    multiplicities = tuple(int(value) for value in multiplicities)
    ratio = tuple(int(value) for value in ratio)
    states = {(0,) * len(ratio)}
    for multiplicity in multiplicities:
        updated_states = set()
        for counts in states:
            for element_index in range(len(ratio)):
                updated = list(counts)
                updated[element_index] += multiplicity
                updated_states.add(tuple(updated))
        states = updated_states
    return any(_is_complete(counts, ratio) for counts in states)


def _sample_state(composition, atoms, wyckoff, spacegroup):
    elements, ratio = reduced_ratio(composition)
    counts = [0] * len(elements)
    element_index = {element: index for index, element in enumerate(elements)}
    previous_w = 0
    for atom, w in zip(np.asarray(atoms), np.asarray(wyckoff)):
        atom = int(atom)
        w = int(w)
        if atom <= 0 or w <= 0:
            continue
        previous_w = w
        if atom in element_index:
            counts[element_index[atom]] += int(_MULTIPLICITIES[spacegroup - 1, w])
    return elements, ratio, tuple(counts), previous_w


def wyckoff_reachability_mask(
    composition,
    atoms,
    wyckoff,
    spacegroup,
    remaining_after_current,
    wyck_types=28,
    max_atoms=512,
):
    """Mask Wyckoff choices that have no feasible element assignment."""
    elements, ratio, counts, previous_w = _sample_state(
        composition, atoms, wyckoff, int(spacegroup)
    )
    allowed = np.zeros(int(wyck_types), dtype=bool)
    if not ratio:
        allowed[:] = True
        return allowed

    allowed[0] = _is_complete(counts, ratio)
    element_order = range(len(elements))
    for w in _next_wyckoff_indices(int(spacegroup), previous_w):
        if w >= wyck_types:
            continue
        multiplicity = int(_MULTIPLICITIES[int(spacegroup) - 1, w])
        for element_index in element_order:
            updated = list(counts)
            updated[element_index] += multiplicity
            if sum(updated) > max_atoms:
                continue
            if _can_finish(
                int(spacegroup),
                ratio,
                w,
                tuple(updated),
                int(remaining_after_current),
                int(max_atoms),
            ):
                allowed[w] = True
                break
    return allowed


def atom_reachability_mask(
    composition,
    atoms,
    wyckoff,
    spacegroup,
    current_w,
    remaining_after_current,
    atom_types=119,
    max_atoms=512,
):
    """Mask element choices that cannot be completed after the current site."""
    elements, ratio, counts, _ = _sample_state(
        composition, atoms, wyckoff, int(spacegroup)
    )
    allowed = np.zeros(int(atom_types), dtype=bool)
    current_w = int(current_w)
    if not ratio:
        allowed[:] = True
        return allowed
    if current_w == 0:
        allowed[0] = _is_complete(counts, ratio)
        return allowed

    multiplicity = int(_MULTIPLICITIES[int(spacegroup) - 1, current_w])
    for element_index, element in enumerate(elements):
        updated = list(counts)
        updated[element_index] += multiplicity
        if sum(updated) > max_atoms:
            continue
        allowed[element] = _can_finish(
            int(spacegroup),
            ratio,
            current_w,
            tuple(updated),
            int(remaining_after_current),
            int(max_atoms),
        )
    return allowed


def batched_wyckoff_reachability_mask(
    composition,
    atoms,
    wyckoff,
    spacegroups,
    remaining_after_current,
    wyck_types,
    max_atoms,
):
    return np.stack(
        [
            wyckoff_reachability_mask(
                composition,
                atom_row,
                wyckoff_row,
                group,
                remaining_after_current,
                wyck_types,
                max_atoms,
            )
            for atom_row, wyckoff_row, group in zip(atoms, wyckoff, spacegroups)
        ]
    )


def trajectory_reachability_masks(
    composition,
    atoms,
    wyckoff,
    spacegroup,
    wyck_types=28,
    atom_types=119,
    max_atoms=512,
):
    """Rebuild the per-step masks used to sample one completed trajectory."""
    atoms = np.asarray(atoms, dtype=np.int32)
    wyckoff = np.asarray(wyckoff, dtype=np.int32)
    _, ratio = reduced_ratio(composition)
    n_max = len(atoms)
    wyckoff_masks = []
    atom_masks = []

    for i in range(n_max):
        prefix_atoms = atoms.copy()
        prefix_wyckoff = wyckoff.copy()
        prefix_atoms[i:] = 0
        prefix_wyckoff[i:] = 0
        remaining_real_sites = max(n_max - i - 2, 0)

        wyckoff_mask = wyckoff_reachability_mask(
            composition,
            prefix_atoms,
            prefix_wyckoff,
            spacegroup,
            remaining_real_sites,
            wyck_types,
            max_atoms,
        )
        if ratio and i == n_max - 1:
            wyckoff_mask[1:] = False
        wyckoff_masks.append(wyckoff_mask)

        atom_masks.append(
            atom_reachability_mask(
                composition,
                prefix_atoms,
                prefix_wyckoff,
                spacegroup,
                wyckoff[i],
                remaining_real_sites,
                atom_types,
                max_atoms,
            )
        )

    return (
        spacegroup_reachability_mask(composition, n_max - 1, max_atoms),
        np.stack(wyckoff_masks),
        np.stack(atom_masks),
    )


def batched_atom_reachability_mask(
    composition,
    atoms,
    wyckoff,
    spacegroups,
    current_w,
    remaining_after_current,
    atom_types,
    max_atoms,
):
    return np.stack([
        atom_reachability_mask(
            composition,
            atom_row,
            wyckoff_row,
            group,
            w,
            remaining_after_current,
            atom_types,
            max_atoms,
        )
        for atom_row, wyckoff_row, group, w in zip(
            atoms, wyckoff, spacegroups, current_w
        )
    ])
