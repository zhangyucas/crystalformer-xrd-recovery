#!/usr/bin/env python
"""Evaluate one independent PPO sample pool with common XRD and structure metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

from crystalformer.reinforce.xrd import (
    cosine_similarity,
    make_xrd_reward_fn,
    peak_match_similarity,
    simulate_structure_pattern,
)
from xrd_recovery import _config, _metadata_for


def _bool(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sampling-csv", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    candidates = Path(args.candidates).resolve()
    ground_truth = Structure.from_file(args.ground_truth)
    config = _config(argparse.Namespace(), _metadata_for(target_path))
    _, reward = make_xrd_reward_fn(
        target=target_path,
        wavelength=config.wavelength,
        two_theta_range=config.two_theta_range,
        grid_step=config.grid_step,
        profile=config.profile,
        fwhm=config.fwhm,
        eta=config.eta,
        target_is_peaks=config.target_is_peaks,
        peak_smoothing=config.peak_smoothing,
        peak_min_width=config.peak_min_width,
    )
    grid = reward.grid
    target_curve = reward.target_curve
    calculator = XRDCalculator(wavelength=config.wavelength)
    matcher = StructureMatcher()
    target_composition = Composition(args.formula).reduced_composition

    with Path(args.sampling_csv).open(newline="") as handle:
        sampling = {int(row["sample_index"]): row for row in csv.DictReader(handle)}

    rows = []
    for sample_index, sample in sorted(sampling.items()):
        cif_path = candidates / f"sample_{sample_index:04d}.cif"
        row = {
            "sample_index": sample_index,
            "formula_match_sampler": int(_bool(sample.get("formula_match"))),
            "geometry_precheck_sampler": int(_bool(sample.get("geometry_precheck"))),
            "cif_written": int(cif_path.is_file()),
            "formula_match_cif": 0,
            "peak_score": 0.0,
            "peak_penalized_score": 0.0,
            "peak_scale": "",
            "scale_penalty": "",
            "cosine_score": 0.0,
            "structure_match": 0,
            "num_sites": "",
            "matched_peaks": "",
            "target_peaks": "",
            "candidate_peaks": "",
            "error": "",
        }
        try:
            if not cif_path.is_file():
                raise FileNotFoundError("candidate CIF was not written")
            structure = Structure.from_file(cif_path)
            row["formula_match_cif"] = int(
                structure.composition.reduced_composition == target_composition
            )
            row["num_sites"] = len(structure)
            _, curve = simulate_structure_pattern(structure, calculator, config, grid)
            peak = peak_match_similarity(curve, target_curve, grid, config)
            peak_penalized = peak_match_similarity(
                curve, target_curve, grid, config, penalize_scale=True
            )
            row["peak_score"] = peak.score
            row["peak_penalized_score"] = peak_penalized.score
            row["peak_scale"] = peak_penalized.scale
            row["scale_penalty"] = peak_penalized.scale_penalty
            row["cosine_score"] = cosine_similarity(curve, target_curve)
            row["structure_match"] = int(matcher.fit(structure, ground_truth))
            row["matched_peaks"] = peak.matched_peaks
            row["target_peaks"] = peak.target_peaks
            row["candidate_peaks"] = peak.candidate_peaks
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if not row["error"]]
    summary = {
        "samples": len(rows),
        "formula_match_sampler": sum(row["formula_match_sampler"] for row in rows),
        "geometry_precheck_sampler": sum(row["geometry_precheck_sampler"] for row in rows),
        "cifs_written": sum(row["cif_written"] for row in rows),
        "formula_match_cif": sum(row["formula_match_cif"] for row in rows),
        "valid_xrd": len(valid),
        "errors": len(rows) - len(valid),
        "zero_peak_scores": sum(float(row["peak_score"]) == 0.0 for row in rows),
        "zero_cosine_scores": sum(float(row["cosine_score"]) == 0.0 for row in rows),
        "mean_peak_score": float(np.mean([row["peak_score"] for row in rows])),
        "mean_peak_penalized_score": float(
            np.mean([row["peak_penalized_score"] for row in rows])
        ),
        "mean_cosine_score": float(np.mean([row["cosine_score"] for row in rows])),
        "best_peak_score": max(float(row["peak_score"]) for row in rows),
        "best_peak_penalized_score": max(
            float(row["peak_penalized_score"]) for row in rows
        ),
        "best_cosine_score": max(float(row["cosine_score"]) for row in rows),
        "structure_matches": sum(row["structure_match"] for row in rows),
        "top10_peak_structure_matches": sum(
            row["structure_match"]
            for row in sorted(rows, key=lambda item: float(item["peak_score"]), reverse=True)[:10]
        ),
        "top10_peak_penalized_structure_matches": sum(
            row["structure_match"]
            for row in sorted(
                rows,
                key=lambda item: float(item["peak_penalized_score"]),
                reverse=True,
            )[:10]
        ),
        "top10_cosine_structure_matches": sum(
            row["structure_match"]
            for row in sorted(rows, key=lambda item: float(item["cosine_score"]), reverse=True)[:10]
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
