#!/usr/bin/env python
"""Primitive-normalized coverage audit and stratified candidate selection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def primitive_structure(structure: Structure, symprec: float = 0.1) -> Structure:
    """Return a symmetry-derived primitive cell with a robust fallback."""

    try:
        primitive = SpacegroupAnalyzer(structure, symprec=symprec).find_primitive()
    except Exception:
        primitive = None
    return primitive or structure.get_primitive_structure(tolerance=symprec)


def primitive_space_group(structure: Structure, symprec: float = 0.1) -> int:
    try:
        return int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
    except Exception:
        return 0


def lattice_is_close(candidate: Structure, target: Structure, length_tol: float, angle_tol: float) -> bool:
    candidate_lengths = np.sort(np.asarray(candidate.lattice.abc, dtype=float))
    target_lengths = np.sort(np.asarray(target.lattice.abc, dtype=float))
    length_error = np.max(np.abs(np.log(candidate_lengths / target_lengths)))
    candidate_angles = np.sort(np.asarray(candidate.lattice.angles, dtype=float))
    target_angles = np.sort(np.asarray(target.lattice.angles, dtype=float))
    return bool(length_error <= length_tol and np.max(np.abs(candidate_angles - target_angles)) <= angle_tol)


def _score_lookup(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {}
    for row in rows:
        score_text = next(
            (
                row[name]
                for name in (
                    "peak_penalized_score", "peak_penalized", "xrd_similarity",
                    "peak_score", "score",
                )
                if row.get(name) not in (None, "")
            ),
            None,
        )
        if score_text is None:
            continue
        keys = [row.get(name) for name in ("path", "cif_path", "candidate", "sample")]
        if row.get("sample_index") not in (None, ""):
            index = int(row["sample_index"])
            keys.extend((str(index), f"sample_{index:04d}.cif"))
        for key in keys:
            if key:
                scores[str(key)] = float(score_text)
                scores[Path(str(key)).name] = float(score_text)
    return scores


def audit_candidates(
    candidate_paths: list[Path],
    target: Structure,
    *,
    formula: str | None = None,
    scores: dict[str, float] | None = None,
    max_variants_per_stratum: int = 3,
    selection_size: int | None = None,
    length_tol: float = 0.15,
    angle_tol: float = 5.0,
    symprec: float = 0.1,
) -> tuple[list[dict], dict]:
    if max_variants_per_stratum <= 0:
        raise ValueError("max_variants_per_stratum must be positive")
    scores = scores or {}
    target_primitive = primitive_structure(target, symprec)
    target_sg = primitive_space_group(target_primitive, symprec)
    target_composition = (
        Composition(formula).reduced_composition
        if formula is not None
        else target_primitive.composition.reduced_composition
    )
    matcher = StructureMatcher(primitive_cell=False, scale=True)
    fixed_lattice_matcher = StructureMatcher(primitive_cell=False, scale=False)
    strata: dict[tuple[int, int], list[tuple[dict, Structure]]] = defaultdict(list)
    rows = []

    for order, path in enumerate(candidate_paths):
        row = {
            "path": str(path), "input_order": order, "valid": 0,
            "formula_match": 0, "primitive_space_group": "",
            "primitive_sites": "", "target_sg": 0, "lattice_close": 0,
            "target_lattice_match": 0, "structure_match": 0,
            "duplicate": 0, "selected": 0, "score": "", "error": "",
        }
        try:
            primitive = primitive_structure(Structure.from_file(path), symprec)
            sg = primitive_space_group(primitive, symprec)
            row.update({
                "valid": 1,
                "formula_match": int(primitive.composition.reduced_composition == target_composition),
                "primitive_space_group": sg,
                "primitive_sites": len(primitive),
                "target_sg": int(sg == target_sg),
                "lattice_close": int(lattice_is_close(primitive, target_primitive, length_tol, angle_tol)),
                "structure_match": int(matcher.fit(primitive, target_primitive)),
            })
            score = scores.get(str(path), scores.get(path.name))
            if score is not None:
                row["score"] = score
            try:
                injected = Structure(
                    target_primitive.lattice,
                    primitive.species,
                    primitive.frac_coords,
                    coords_are_cartesian=False,
                )
                row["target_lattice_match"] = int(
                    fixed_lattice_matcher.fit(injected, target_primitive)
                )
            except Exception:
                pass

            stratum = (sg, len(primitive))
            if row["formula_match"]:
                existing = strata[stratum]
                if any(matcher.fit(primitive, kept) for _, kept in existing):
                    row["duplicate"] = 1
                elif len(existing) < max_variants_per_stratum:
                    existing.append((row, primitive))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    queues = {
        key: deque(sorted(items, key=lambda item: float(item[0]["score"] or -np.inf), reverse=True))
        for key, items in sorted(strata.items())
        if items
    }
    selected = []
    limit = sum(len(queue) for queue in queues.values()) if selection_size is None else selection_size
    while queues and len(selected) < limit:
        for key in list(queues):
            if len(selected) >= limit:
                break
            row, _ = queues[key].popleft()
            row["selected"] = 1
            selected.append(row)
            if not queues[key]:
                del queues[key]

    ranked = sorted(
        (row for row in rows if row["score"] != ""),
        key=lambda row: float(row["score"]), reverse=True,
    )
    first_rank = next((index for index, row in enumerate(ranked, 1) if row["structure_match"]), None)
    input_match_rank = next(
        (index for index, row in enumerate(rows, 1) if row["structure_match"]), None
    )
    selected_match_rank = next(
        (index for index, row in enumerate(selected, 1) if row["structure_match"]), None
    )
    formula_rows = [row for row in rows if row["formula_match"]]
    target_sg_rows = [row for row in formula_rows if row["target_sg"]]
    summary = {
        "input_candidates": len(rows),
        "valid_candidates": sum(row["valid"] for row in rows),
        "formula_only_pool": len(formula_rows),
        "target_space_group_subset": len(target_sg_rows),
        "target_sg_lattice_proximity_subset": sum(row["lattice_close"] for row in target_sg_rows),
        "target_lattice_injected_matches": sum(row["target_lattice_match"] for row in formula_rows),
        "structure_match_oracle_hits": sum(row["structure_match"] for row in formula_rows),
        "first_structure_match_score_rank": first_rank,
        "first_structure_match_input_rank": input_match_rank,
        "first_structure_match_selected_rank": selected_match_rank,
        "primitive_strata": len(strata),
        "near_duplicates_removed": sum(row["duplicate"] for row in rows),
        "selected_candidates": len(selected),
        "target_primitive_space_group": target_sg,
        "target_primitive_sites": len(target_primitive),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sampling-csv")
    parser.add_argument("--formula")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-variants-per-stratum", type=int, default=3)
    parser.add_argument("--selection-size", type=int)
    parser.add_argument("--length-tol", type=float, default=0.15)
    parser.add_argument("--angle-tol", type=float, default=5.0)
    args = parser.parse_args()

    candidate_root = Path(args.candidates)
    paths = sorted(candidate_root.rglob("*.cif")) if candidate_root.is_dir() else [candidate_root]
    rows, summary = audit_candidates(
        paths,
        Structure.from_file(args.target),
        formula=args.formula,
        scores=_score_lookup(Path(args.sampling_csv) if args.sampling_csv else None),
        max_variants_per_stratum=args.max_variants_per_stratum,
        selection_size=args.selection_size,
        length_tol=args.length_tol,
        angle_tol=args.angle_tol,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
