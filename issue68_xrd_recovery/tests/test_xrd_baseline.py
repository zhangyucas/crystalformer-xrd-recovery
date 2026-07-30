import numpy as np

from crystalformer.reinforce.xrd_baseline import (
    RandomMoveConfig,
    formula_species,
    run_random_move,
)


def test_formula_species_uses_reduced_stoichiometry():
    assert formula_species("Si2O4") == ["Si", "O", "O"]


def test_random_move_is_bounded_and_reproducible():
    config = RandomMoveConfig(evaluations=24, restarts=3, seed=17)

    def score(structure):
        # A cheap deterministic stand-in for the XRD scorer.
        return float(1.0 / (1.0 + abs(structure.volume - 45.0)))

    first = run_random_move("Si", score, config)
    second = run_random_move("Si", score, config)
    assert first["evaluations"] <= config.evaluations
    assert first["attempts"] <= config.evaluations
    assert len(first["records"]) > 0
    assert first["best_score"] == second["best_score"]
    assert first["best_structure"].composition.reduced_formula == "Si"


def test_random_move_preserves_multielement_formula():
    config = RandomMoveConfig(evaluations=8, restarts=1, seed=2)
    result = run_random_move("NaCl", lambda structure: float(np.clip(structure.volume / 100.0, 0, 1)), config)
    assert result["best_structure"].composition.reduced_formula == "NaCl"
