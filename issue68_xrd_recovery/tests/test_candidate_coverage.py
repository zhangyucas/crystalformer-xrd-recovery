from pymatgen.core import Lattice, Structure

from issue68_xrd_recovery.scripts.candidate_coverage import audit_candidates, main


def _write(structure, path):
    structure.to(filename=str(path))
    return path


def test_coverage_audit_deduplicates_and_round_robins_strata(tmp_path):
    target = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    paths = [
        _write(target, tmp_path / "target_copy.cif"),
        _write(target.copy(), tmp_path / "duplicate.cif"),
        _write(
            Structure(
                Lattice.from_parameters(4.2, 4.5, 4.9, 78, 83, 87),
                ["Si", "Si"],
                [[0.1, 0.2, 0.3], [0.37, 0.41, 0.52]],
            ),
            tmp_path / "p1.cif",
        ),
        _write(Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]), tmp_path / "wrong_formula.cif"),
    ]
    scores = {path.name: score for path, score in zip(paths, (0.9, 0.8, 0.7, 0.6))}
    rows, summary = audit_candidates(
        paths, target, formula="Si", scores=scores,
        max_variants_per_stratum=1, selection_size=3,
    )

    assert summary["input_candidates"] == 4
    assert summary["formula_only_pool"] == 3
    assert summary["structure_match_oracle_hits"] == 2
    assert summary["first_structure_match_score_rank"] == 1
    assert summary["near_duplicates_removed"] >= 1
    assert summary["selected_candidates"] <= 3
    assert any(row["selected"] for row in rows)


def test_target_lattice_injection_is_reported_separately(tmp_path):
    target = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    scaled = Structure(Lattice.cubic(5.0), ["Si"], [[0, 0, 0]])
    rows, summary = audit_candidates(
        [_write(scaled, tmp_path / "scaled.cif")], target, formula="Si"
    )
    assert summary["formula_only_pool"] == 1
    assert summary["target_lattice_injected_matches"] == 1
    assert rows[0]["target_lattice_match"] == 1


def test_cli_recursively_reads_seed_subdirectories(tmp_path, monkeypatch):
    target = _write(
        Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]]),
        tmp_path / "target.cif",
    )
    nested = tmp_path / "candidates" / "seed2"
    nested.mkdir(parents=True)
    _write(Structure(Lattice.cubic(3.6), ["Si"], [[0, 0, 0]]), nested / "sample.cif")
    output = tmp_path / "coverage.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "candidate_coverage", "--target", str(target),
            "--candidates", str(tmp_path / "candidates"),
            "--output", str(output),
        ],
    )
    main()
    assert len(output.read_text().splitlines()) == 2
