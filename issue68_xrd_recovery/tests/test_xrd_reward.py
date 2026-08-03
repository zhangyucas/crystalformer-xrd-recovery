import csv
import json

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from crystalformer.reinforce.xrd import (
    broaden_peaks,
    cosine_similarity,
    extract_pattern_peaks,
    load_xrd_pattern,
    make_xrd_reward_fn,
    peak_match_similarity,
    XRDConfig,
    save_pattern,
    structure_from_GLXYZAW,
)
from issue68_xrd_recovery.scripts.xrd_recovery import build_parser as recovery_parser


def _p1_sample(lattice=3.5, atomic_number=14):
    return (
        np.asarray(1),
        np.asarray([lattice, lattice, lattice, 90.0, 90.0, 90.0]),
        np.asarray([[0.0, 0.0, 0.0], [1e10, 1e10, 1e10]]),
        np.asarray([atomic_number, 0]),
        np.asarray([1, 0]),
    )


def test_structure_conversion_and_self_similarity():
    sample = _p1_sample()
    structure = structure_from_GLXYZAW(*sample)
    assert structure.composition.reduced_formula == "Si"

    reward_fn, batch_reward_fn = make_xrd_reward_fn(
        target_structure=structure,
        grid_step=0.2,
    )
    assert reward_fn(sample) > 0.999999
    scores = batch_reward_fn(tuple(np.stack([value, value]) for value in sample))
    np.testing.assert_allclose(np.asarray(scores), [1.0, 1.0], atol=1e-6)


def test_reward_tolerates_global_lattice_scale_change():
    target = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    reward_fn, _ = make_xrd_reward_fn(target_structure=target, grid_step=0.2)
    assert reward_fn(_p1_sample(lattice=4.2)) > 0.99


def test_profiles_and_cosine_are_finite():
    grid = np.linspace(5.0, 90.0, 500)
    gaussian = broaden_peaks([20.0, 40.0], [1.0, 0.5], grid, fwhm=0.2)
    pvoigt = broaden_peaks(
        [20.0, 40.0], [1.0, 0.5], grid,
        profile="pseudo-voigt", fwhm=0.2, eta=0.3,
    )
    assert np.isfinite(gaussian).all()
    assert np.isfinite(pvoigt).all()
    assert cosine_similarity(gaussian, gaussian) == 1.0
    assert cosine_similarity(np.zeros_like(grid), gaussian) == 0.0


def test_peak_score_tolerates_global_lattice_scale():
    grid = np.linspace(5.0, 90.0, 1701)
    config = XRDConfig(profile="pseudo-voigt", fwhm=0.5)
    target = broaden_peaks(
        [20.0, 32.0, 47.0, 65.0], [1.0, 0.7, 0.4, 0.2], grid,
        profile=config.profile, fwhm=config.fwhm,
    )
    # A uniform q-space expansion represents a global lattice-scale error.
    wavelength = 1.5406
    q = 4 * np.pi * np.sin(np.deg2rad(np.array([20.0, 32.0, 47.0, 65.0]) / 2)) / wavelength
    shifted_q = q / 1.12
    shifted_theta = 2 * np.rad2deg(np.arcsin(shifted_q * wavelength / (4 * np.pi)))
    candidate = broaden_peaks(
        shifted_theta, [1.0, 0.7, 0.4, 0.2], grid,
        profile=config.profile, fwhm=config.fwhm,
    )
    result = peak_match_similarity(candidate, target, grid, config)
    assert result.score > 0.98
    assert result.scale == pytest.approx(1.12, abs=0.02)


def test_peak_score_penalizes_missing_and_extra_peaks():
    grid = np.linspace(5.0, 90.0, 1701)
    config = XRDConfig(profile="pseudo-voigt", fwhm=0.5)
    target = broaden_peaks(
        [20.0, 35.0, 50.0, 70.0], [1.0, 0.8, 0.6, 0.4], grid,
        profile=config.profile, fwhm=config.fwhm,
    )
    exact = broaden_peaks(
        [20.0, 35.0, 50.0, 70.0], [1.0, 0.8, 0.6, 0.4], grid,
        profile=config.profile, fwhm=config.fwhm,
    )
    wrong = broaden_peaks(
        [20.0, 35.0, 43.0, 58.0, 70.0, 82.0],
        [1.0, 0.8, 0.7, 0.6, 0.4, 0.3], grid,
        profile=config.profile, fwhm=config.fwhm,
    )
    exact_score = peak_match_similarity(exact, target, grid, config).score
    wrong_score = peak_match_similarity(wrong, target, grid, config).score
    assert exact_score > 0.99
    assert wrong_score < exact_score - 0.2


def test_peak_extraction_smoothing_removes_noise_spikes():
    grid = np.linspace(5.0, 90.0, 1701)
    clean = broaden_peaks([31.75, 45.5, 56.5, 66.3, 75.35, 84.05],
                          [1.0, 0.66, 0.21, 0.09, 0.25, 0.18], grid,
                          profile="pseudo-voigt", fwhm=0.5)
    rng = np.random.default_rng(731)
    noisy = clean + rng.normal(0.0, 0.02, grid.size)
    noisy = np.maximum(noisy, 0.0)
    config = XRDConfig(
        grid_step=0.05,
        profile="pseudo-voigt",
        fwhm=0.5,
        peak_smoothing=0.10,
    )
    unsmoothed = XRDConfig(**{**config.__dict__, "peak_smoothing": 0.0})
    assert len(extract_pattern_peaks(grid, noisy, unsmoothed)[0]) > 6
    assert len(extract_pattern_peaks(grid, noisy, config)[0]) == 6


def test_peak_width_rejects_single_point_spike():
    grid = np.linspace(5.0, 90.0, 1701)
    curve = broaden_peaks([30.0], [1.0], grid, fwhm=0.5)
    curve[np.argmin(np.abs(grid - 60.0))] = 1.0
    config = XRDConfig(grid_step=0.05, peak_smoothing=0.10, peak_min_width=0.05)
    positions, _ = extract_pattern_peaks(grid, curve, config)
    np.testing.assert_allclose(positions, [30.0], atol=0.05)


def test_weak_extra_peak_has_smaller_penalty_than_strong_extra_peak():
    grid = np.linspace(5.0, 90.0, 1701)
    config = XRDConfig(profile="pseudo-voigt", fwhm=0.5)
    target = broaden_peaks([20.0, 40.0, 60.0], [1.0, 0.8, 0.6], grid,
                           profile=config.profile, fwhm=config.fwhm)
    weak_extra = broaden_peaks([20.0, 40.0, 60.0, 75.0], [1.0, 0.8, 0.6, 0.1], grid,
                               profile=config.profile, fwhm=config.fwhm)
    strong_extra = broaden_peaks([20.0, 40.0, 60.0, 75.0], [1.0, 0.8, 0.6, 1.0], grid,
                                 profile=config.profile, fwhm=config.fwhm)
    weak_score = peak_match_similarity(weak_extra, target, grid, config).score
    strong_score = peak_match_similarity(strong_extra, target, grid, config).score
    assert weak_score > strong_score


def test_csv_target_and_score_output(tmp_path):
    target = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    target_path = tmp_path / "target.csv"
    reward_target, batch_target = make_xrd_reward_fn(target_structure=target, grid_step=0.2)
    save_pattern(target_path, batch_target.grid, batch_target.target_curve)

    x, y = load_xrd_pattern(target_path)
    assert len(x) == len(y) == len(batch_target.grid)
    reward_file, batch_file = make_xrd_reward_fn(target=target_path, grid_step=0.2)
    sample = _p1_sample()
    assert reward_file(sample) > 0.999999

    output = batch_file(
        tuple(np.stack([value]) for value in sample),
        str(tmp_path),
        3,
    )
    assert np.asarray(output).shape == (1,)
    with (tmp_path / "xrd_scores_3.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["sample", "xrd_similarity"]
    assert len(rows) == 2
    assert (tmp_path / "xrd_scores_3_candidates" / "sample_0000.cif").is_file()


def test_single_row_csv_target_is_scalar_safe(tmp_path):
    target_path = tmp_path / "single_peak.csv"
    target_path.write_text("two_theta,intensity\n20,1\n")
    positions, intensities = load_xrd_pattern(target_path)
    assert positions.shape == (1,)
    assert intensities.shape == (1,)
    assert positions[0] == 20.0
    assert intensities[0] == 1.0


def test_common_two_theta_header_aliases(tmp_path):
    target_path = tmp_path / "aliases.csv"
    target_path.write_text("2theta,counts\n20,1\n30,0.5\n")
    positions, intensities = load_xrd_pattern(target_path)
    np.testing.assert_allclose(positions, [20.0, 30.0])
    np.testing.assert_allclose(intensities, [1.0, 0.5])


def test_invalid_sample_returns_invalid_reward():
    target = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    reward_fn, _ = make_xrd_reward_fn(target_structure=target, grid_step=0.2)
    invalid = _p1_sample()
    invalid = (invalid[0], invalid[1], invalid[2], np.asarray([0, 0]), invalid[4])
    assert reward_fn(invalid) == 0.0


def test_candidate_evaluation_summarizes_structure_matches(tmp_path):
    target = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    target.to(filename=str(tmp_path / "target.cif"))
    target.to(filename=str(candidates / "match.cif"))
    Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(
        filename=str(candidates / "mismatch.cif")
    )
    output = tmp_path / "evaluation.csv"
    args = recovery_parser().parse_args([
        "evaluate",
        "--target", str(tmp_path / "target.cif"),
        "--candidates", str(candidates),
        "--ground-truth", str(tmp_path / "target.cif"),
        "--output", str(output),
        "--top-k", "1",
    ])
    args.func(args)

    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert summary["structure_match_count"] == 1
    assert summary["structure_match_rate"] == 0.5
    assert summary["first_structure_match_rank"] is not None
    assert summary["top_k_structure_match"] == (summary["first_structure_match_rank"] == 1)
