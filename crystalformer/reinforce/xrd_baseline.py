"""Small, deterministic random-move baselines for powder-XRD recovery.

The baseline is intentionally independent of CrystalFormer.  It samples a
P1 structure with the requested composition, proposes local coordinate or
lattice moves, and accepts them with a Metropolis rule.  A bounded evaluation
budget and serial execution make it suitable for a small CPU/WSL machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import exp, gcd, lcm
from typing import Callable, Sequence

import numpy as np
from pymatgen.core import Composition, Lattice, Structure


@dataclass(frozen=True)
class RandomMoveConfig:
    """Controls one bounded random-move search."""

    evaluations: int = 500
    restarts: int = 4
    initial_temperature: float = 0.05
    final_temperature: float = 0.002
    coordinate_sigma: float = 0.04
    lattice_sigma: float = 0.04
    min_distance: float = 0.65
    volume_per_atom_min: float = 10.0
    volume_per_atom_max: float = 35.0
    length_min: float = 2.0
    length_max: float = 20.0
    angle_min: float = 55.0
    angle_max: float = 125.0
    coordinate_move_probability: float = 0.8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.evaluations <= 0:
            raise ValueError("evaluations must be positive")
        if self.restarts <= 0:
            raise ValueError("restarts must be positive")
        if self.initial_temperature <= 0 or self.final_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature cannot exceed initial_temperature")
        if self.coordinate_sigma <= 0 or self.lattice_sigma <= 0:
            raise ValueError("move widths must be positive")
        if self.min_distance < 0:
            raise ValueError("min_distance cannot be negative")
        if self.volume_per_atom_min <= 0 or self.volume_per_atom_max < self.volume_per_atom_min:
            raise ValueError("invalid volume-per-atom bounds")
        if self.length_min <= 0 or self.length_max < self.length_min:
            raise ValueError("invalid lattice-length bounds")
        if not 0.0 < self.coordinate_move_probability <= 1.0:
            raise ValueError("coordinate_move_probability must be in (0, 1]")
        if self.angle_min <= 0 or self.angle_max >= 180 or self.angle_max < self.angle_min:
            raise ValueError("invalid lattice-angle bounds")

    def as_dict(self) -> dict:
        return asdict(self)


def formula_species(formula: str, *, max_atoms: int | None = None) -> list[str]:
    """Expand a chemical formula into a deterministic list of element symbols.

    CrystalFormer's conditional sampler works with reduced compositions.  The
    same convention is used here, so ``Si2O4`` and ``SiO2`` create the same
    three-site P1 baseline.  Fractional amounts are accepted when they can be
    represented by a small integer ratio.
    """

    composition = Composition(formula).reduced_composition
    amounts = []
    denominator_lcm = 1
    for element in composition.elements:
        amount = Fraction(str(float(composition[element]))).limit_denominator(1000)
        if amount <= 0:
            continue
        amounts.append((str(element), amount))
        denominator_lcm = lcm(denominator_lcm, amount.denominator)
    if not amounts:
        raise ValueError(f"formula has no positive element amounts: {formula}")

    counts = [int(amount * denominator_lcm) for _, amount in amounts]
    common = 0
    for count in counts:
        common = gcd(common, count)
    counts = [count // max(common, 1) for count in counts]
    total = sum(counts)
    if max_atoms is not None and total > max_atoms:
        raise ValueError(
            f"reduced formula {formula} needs {total} atoms, exceeding max_atoms={max_atoms}"
        )
    return [symbol for (symbol, _), count in zip(amounts, counts) for _ in range(count)]


def _random_lattice(
    rng: np.random.Generator, n_atoms: int, config: RandomMoveConfig
) -> Lattice:
    """Draw a valid triclinic lattice with a bounded volume per atom."""

    target_volume = rng.uniform(
        config.volume_per_atom_min, config.volume_per_atom_max
    ) * max(n_atoms, 1)
    for _ in range(100):
        lengths = rng.uniform(config.length_min, config.length_max, size=3)
        angles = rng.uniform(config.angle_min, config.angle_max, size=3)
        try:
            lattice = Lattice.from_parameters(*lengths, *angles)
        except Exception:
            continue
        if lattice.volume <= 1e-8:
            continue
        scale = (target_volume / lattice.volume) ** (1.0 / 3.0)
        scaled_lengths = lengths * scale
        if np.all((scaled_lengths >= config.length_min) & (scaled_lengths <= config.length_max)):
            return Lattice.from_parameters(*scaled_lengths, *angles)

    # A cubic fallback keeps a pathological set of bounds from aborting a
    # whole baseline sweep.
    side = float(np.clip(target_volume ** (1.0 / 3.0), config.length_min, config.length_max))
    return Lattice.cubic(side)


def _minimum_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return float("inf")
    distances = np.asarray(structure.distance_matrix, dtype=np.float64)
    distances[distances <= 1e-10] = np.inf
    return float(np.min(distances))


def random_structure(
    species: Sequence[str], rng: np.random.Generator, config: RandomMoveConfig
) -> Structure:
    """Generate one composition-preserving P1 structure."""

    species = list(species)
    if not species:
        raise ValueError("species must contain at least one atom")
    best = None
    best_distance = -np.inf
    for _ in range(80):
        lattice = _random_lattice(rng, len(species), config)
        coords = rng.random((len(species), 3))
        try:
            candidate = Structure(lattice, species, coords, coords_are_cartesian=False)
        except Exception:
            continue
        distance = _minimum_distance(candidate)
        if distance > best_distance:
            best, best_distance = candidate, distance
        if distance >= config.min_distance:
            return candidate
    if best is None:
        raise RuntimeError("could not construct a random structure")
    return best


def _propose(
    current: Structure,
    rng: np.random.Generator,
    config: RandomMoveConfig,
) -> tuple[Structure | None, str]:
    """Propose one bounded local move, preserving species and site count."""

    if rng.random() < config.coordinate_move_probability:
        coords = np.asarray(current.frac_coords, dtype=np.float64).copy()
        index = int(rng.integers(len(coords)))
        coords[index] = (coords[index] + rng.normal(0.0, config.coordinate_sigma, 3)) % 1.0
        move = "coordinate"
        lattice = current.lattice
    else:
        parameters = np.asarray(current.lattice.parameters, dtype=np.float64)
        if rng.random() < 0.7:
            index = int(rng.integers(3))
            parameters[index] *= exp(float(rng.normal(0.0, config.lattice_sigma)))
        else:
            index = 3 + int(rng.integers(3))
            parameters[index] += float(rng.normal(0.0, config.lattice_sigma * 20.0))
        if not np.all((parameters[:3] >= config.length_min) & (parameters[:3] <= config.length_max)):
            return None, "lattice"
        parameters[3:] = np.clip(parameters[3:], config.angle_min, config.angle_max)
        try:
            lattice = Lattice.from_parameters(*parameters)
        except Exception:
            return None, "lattice"
        coords = np.asarray(current.frac_coords, dtype=np.float64)
        move = "lattice"

    try:
        candidate = Structure(lattice, current.species, coords, coords_are_cartesian=False)
    except Exception:
        return None, move
    if _minimum_distance(candidate) < config.min_distance:
        return None, move
    return candidate, move


def run_random_move(
    formula: str,
    score_fn: Callable[[Structure], float],
    config: RandomMoveConfig | None = None,
) -> dict:
    """Run a bounded multi-restart random-move Monte Carlo search.

    Returns plain Python/NumPy data so callers can stream it to CSV/JSON
    without retaining every structure.  ``best_structure`` is the only
    structure object retained in the result.
    """

    config = config or RandomMoveConfig()
    species = formula_species(formula)
    rng = np.random.default_rng(config.seed)
    records: list[dict] = []
    restart_results: list[dict] = []
    best_structure: Structure | None = None
    best_score = -np.inf
    evaluations = 0
    attempts = 0

    def finite_score(structure: Structure) -> float:
        try:
            value = float(score_fn(structure))
        except Exception:
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    for restart in range(config.restarts):
        if attempts >= config.evaluations:
            break
        remaining_restarts = config.restarts - restart
        remaining_budget = config.evaluations - attempts
        restart_budget = max(1, remaining_budget // remaining_restarts)
        current = random_structure(species, rng, config)
        current_score = finite_score(current)
        evaluations += 1
        attempts += 1
        restart_best = current_score
        accepted = 1
        records.append({
            "evaluation": evaluations,
            "attempt": attempts,
            "restart": restart,
            "step": 0,
            "score": current_score,
            "best_score": max(best_score, current_score),
            "accepted": True,
            "scored": True,
            "move": "initial",
            "temperature": config.initial_temperature,
        })
        if best_structure is None or current_score > best_score:
            best_score, best_structure = current_score, current.copy()

        for step in range(1, restart_budget):
            if attempts >= config.evaluations:
                break
            attempts += 1
            proposal, move = _propose(current, rng, config)
            temperature_fraction = step / max(restart_budget - 1, 1)
            temperature = config.initial_temperature * (
                config.final_temperature / config.initial_temperature
            ) ** temperature_fraction
            accepted_move = False
            proposal_score = float("-inf")
            if proposal is not None:
                proposal_score = finite_score(proposal)
                evaluations += 1
                delta = proposal_score - current_score
                accepted_move = delta >= 0 or rng.random() < exp(
                    max(-700.0, min(0.0, delta / temperature))
                )
                if accepted_move:
                    current, current_score = proposal, proposal_score
                    accepted += 1
                    restart_best = max(restart_best, current_score)
                    if current_score > best_score:
                        best_score, best_structure = current_score, current.copy()
            else:
                # Invalid proposals consume no simulator call and are recorded
                # so the trace still explains why a restart made little progress.
                proposal_score = current_score

            records.append({
                "evaluation": evaluations,
                "attempt": attempts,
                "restart": restart,
                "step": step,
                "score": proposal_score,
                "best_score": best_score,
                "accepted": accepted_move,
                "scored": proposal is not None,
                "move": move,
                "temperature": temperature,
            })

        restart_records = [row for row in records if row["restart"] == restart]
        restart_results.append({
            "restart": restart,
            "attempts": len(restart_records),
            "evaluations": sum(1 for row in restart_records if row["scored"]),
            "accepted": accepted,
            "best_score": restart_best,
        })

    if best_structure is None:
        raise RuntimeError("random-move search produced no structure")
    return {
        "best_structure": best_structure,
        "best_score": float(best_score),
        "evaluations": evaluations,
        "attempts": attempts,
        "records": records,
        "restarts": restart_results,
        "config": config.as_dict(),
        "formula": formula,
    }


def structure_matches(candidate: Structure, reference: Structure) -> bool:
    """Return a StructureMatcher result without making it a search objective."""

    from pymatgen.analysis.structure_matcher import StructureMatcher

    return bool(StructureMatcher().fit(candidate, reference))


__all__ = [
    "RandomMoveConfig",
    "formula_species",
    "random_structure",
    "run_random_move",
    "structure_matches",
]
