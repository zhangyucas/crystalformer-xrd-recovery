#!/usr/bin/env python
"""Export the official SimXRD code-test ASE database for recovery tests."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor


CU_KA_ANGSTROM = 1.54184


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in connect(args.database).select():
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula(mode="reduce")
        case_dir = output / f"case_{row.id:02d}_{formula}"
        case_dir.mkdir(parents=True, exist_ok=True)
        cif_path = case_dir / "ground_truth.cif"
        has_structure = bool(atoms.pbc.all() and abs(np.linalg.det(atoms.cell.array)) > 1e-8)
        if has_structure:
            structure = AseAtomsAdaptor.get_structure(atoms)
            formula = structure.composition.reduced_formula
            structure.to(filename=str(cif_path))

        d_spacing = np.asarray(ast.literal_eval(row.latt_dis), dtype=float)
        intensity = np.asarray(ast.literal_eval(row.intensity), dtype=float)
        valid = np.isfinite(d_spacing) & np.isfinite(intensity) & (d_spacing > CU_KA_ANGSTROM / 2)
        d_spacing = d_spacing[valid]
        intensity = np.maximum(intensity[valid], 0.0)
        two_theta = np.degrees(2.0 * np.arcsin(CU_KA_ANGSTROM / (2.0 * d_spacing)))
        order = np.argsort(two_theta)
        target_path = case_dir / "target.csv"
        with target_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["two_theta", "intensity"])
            writer.writerows(zip(two_theta[order], intensity[order]))
        metadata = {
            "source": "SimXRD official GitHub code_test/binxrd.db",
            "source_row": row.id,
            "wavelength": CU_KA_ANGSTROM,
            "two_theta_min": float(two_theta.min()),
            "two_theta_max": float(two_theta.max()),
            "grid_step": 0.02,
            "profile": "gaussian",
            "fwhm": 0.1,
            "eta": 0.5,
            "target_is_peaks": False,
            "simxrd_label": ast.literal_eval(row.tager),
            "simxrd_parameters": ast.literal_eval(row.simulation_param),
        }
        (target_path.with_suffix(".csv.json")).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        manifest.append({
            "case": row.id,
            "formula": formula,
            "num_sites": len(atoms),
            "space_group": metadata["simxrd_label"][0],
            "target": str(target_path),
            "ground_truth": str(cif_path) if has_structure else "",
            "structure_match_eligible": has_structure,
        })

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Exported {len(manifest)} official SimXRD cases to {output}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
