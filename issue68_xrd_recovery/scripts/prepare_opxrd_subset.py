#!/usr/bin/env python
"""Extract a small, fully labeled single-phase subset from the opXRD ZIP."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure


def _loads(value):
    while isinstance(value, str):
        value = json.loads(value)
    return value


def _element(symbol: str) -> str:
    match = re.match(r"^([A-Z][a-z]?)", symbol.strip())
    return match.group(1) if match else ""


def _structure(phase: dict) -> Structure | None:
    if not phase.get("lattice") or not phase.get("basis"):
        return None
    lattice_values = phase["lattice"].strip("()[]").split(",")
    if len(lattice_values) != 6:
        return None
    a, b, c, alpha, beta, gamma = map(float, lattice_values)
    sites = [_loads(site) for site in _loads(phase["basis"])]
    species, coords = [], []
    for site in sites:
        symbol = _element(str(site.get("symbol", "")))
        occupancy = float(site.get("occupancy", 1.0))
        if not symbol or occupancy <= 0:
            return None
        species.append({symbol: min(occupancy, 1.0)})
        coords.append([float(site[key]) for key in ("x", "y", "z")])
    if not species:
        return None
    return Structure(
        Lattice.from_parameters(a, b, c, alpha, beta, gamma),
        species,
        coords,
        coords_are_cartesian=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-sites", type=int, default=64)
    parser.add_argument("--max-elements", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    counts = {key: 0 for key in (
        "patterns", "labeled", "single_phase", "full_structure", "eligible", "errors"
    )}
    with zipfile.ZipFile(args.zip) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            counts["patterns"] += 1
            try:
                record = json.loads(archive.read(name))
                label = _loads(record.get("label") or "{}")
                phases = label.get("phases") or []
                if phases:
                    counts["labeled"] += 1
                if len(phases) != 1 or bool(label.get("is_simulated")):
                    continue
                counts["single_phase"] += 1
                phase = _loads(phases[0])
                structure = _structure(phase)
                if structure is None:
                    continue
                counts["full_structure"] += 1
                if len(structure) > args.max_sites or len(structure.composition.elements) > args.max_elements:
                    continue
                if not structure.is_ordered:
                    continue
                counts["eligible"] += 1
                candidates.append((name, record, label, phase, structure))
            except Exception:
                counts["errors"] += 1

    # Favor smaller cells, then cover different formulas/institutions deterministically.
    candidates.sort(key=lambda item: (len(item[4]), item[4].composition.reduced_formula, item[0]))
    selected, formulas, institutions = [], set(), set()
    for item in candidates:
        formula = item[4].composition.reduced_formula
        institution = item[0].split("/", 1)[0]
        if formula in formulas and institution in institutions:
            continue
        selected.append(item)
        formulas.add(formula)
        institutions.add(institution)
        if len(selected) >= args.max_cases:
            break

    manifest = []
    for index, (name, record, label, phase, structure) in enumerate(selected, 1):
        formula = structure.composition.reduced_formula
        case_dir = output / f"case_{index:02d}_{formula}"
        case_dir.mkdir(parents=True, exist_ok=True)
        cif_path = case_dir / "ground_truth.cif"
        structure.to(filename=str(cif_path))
        theta = np.asarray(record["two_theta_values"], dtype=float)
        intensity = np.asarray(record["intensities"], dtype=float)
        finite = np.isfinite(theta) & np.isfinite(intensity)
        theta, intensity = theta[finite], np.maximum(intensity[finite], 0.0)
        target_path = case_dir / "target.csv"
        with target_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["two_theta", "intensity"])
            writer.writerows(zip(theta, intensity))
        xray = _loads(label.get("xray_info") or "{}")
        wavelength = float(xray.get("primary_wavelength") or 1.5406)
        step = float(np.median(np.diff(np.unique(theta))))
        metadata = {
            "source": "opXRD Zenodo record 15298026",
            "source_member": name,
            "wavelength": wavelength,
            "two_theta_min": float(theta.min()),
            "two_theta_max": float(theta.max()),
            "grid_step": step,
            "profile": "pseudo-voigt",
            "fwhm": 0.5,
            "eta": 0.5,
            "target_is_peaks": False,
            "experimental": True,
        }
        target_path.with_suffix(".csv.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        manifest.append({
            "case": index,
            "formula": formula,
            "num_sites": len(structure),
            "space_group_label": phase.get("spacegroup") or "",
            "source_member": name,
            "target": str(target_path),
            "ground_truth": str(cif_path),
        })

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]) if manifest else ["case"])
        writer.writeheader()
        writer.writerows(manifest)
    (output / "scan_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"Extracted {len(manifest)} opXRD cases to {output}")


if __name__ == "__main__":
    main()
