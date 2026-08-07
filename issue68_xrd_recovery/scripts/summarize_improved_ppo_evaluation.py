#!/usr/bin/env python
"""Summarize paired fixed-condition improved PPO evaluations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


INT_METRICS = (
    "samples", "cifs_written", "structure_matches",
    "top10_peak_structure_matches", "top10_peak_penalized_structure_matches",
    "coverage_primitive_strata", "coverage_near_duplicates_removed",
    "coverage_target_space_group_subset",
    "coverage_target_sg_lattice_proximity_subset",
    "coverage_target_lattice_injected_matches",
)
FLOAT_METRICS = (
    "mean_peak_score", "mean_peak_penalized_score", "mean_cosine_score",
)
METHODS = ("peak", "peak_penalized")
EPOCHS = (0, 5, 25)


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=float)
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(float(np.mean(values * np.asarray(signs))))
        extreme += int(permuted >= observed - 1e-12)
        total += 1
    return extreme / total


def aggregate(rows: list[dict]) -> dict:
    result = {"pools": len(rows)}
    for metric in INT_METRICS:
        result[metric] = int(sum(row[metric] for row in rows))
    for metric in FLOAT_METRICS:
        result[metric] = float(np.mean([row[metric] for row in rows]))
    return result


def paired_test(after: np.ndarray, before: np.ndarray) -> dict:
    difference = after - before
    try:
        wilcoxon_p = float(wilcoxon(difference).pvalue)
    except ValueError:
        wilcoxon_p = None
    return {
        "differences": [float(value) for value in difference],
        "mean_difference": float(np.mean(difference)),
        "sum_difference": float(np.sum(difference)),
        "exact_sign_flip_pvalue": exact_sign_flip_pvalue(difference.tolist()),
        "paired_t_pvalue": float(ttest_rel(after, before).pvalue),
        "wilcoxon_pvalue": wilcoxon_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["checkpoint_epoch"] = int(row["checkpoint_epoch"])
        row["training_seed"] = int(row["training_seed"])
        for metric in INT_METRICS:
            row[metric] = int(row[metric])
        for metric in FLOAT_METRICS:
            row[metric] = float(row[metric])
    if len(rows) != 72:
        raise ValueError(f"expected 72 evaluation pools, got {len(rows)}")

    payload: dict = {"overall": {}, "by_formula": {}, "paired": {}}
    for method in METHODS:
        payload["overall"][method] = {}
        for epoch in EPOCHS:
            selected = [
                row for row in rows
                if row["training_method"] == method
                and row["checkpoint_epoch"] == epoch
            ]
            payload["overall"][method][str(epoch)] = aggregate(selected)

    formulas = sorted({row["formula"] for row in rows})
    for formula in formulas:
        payload["by_formula"][formula] = {}
        for method in METHODS:
            payload["by_formula"][formula][method] = {}
            for epoch in EPOCHS:
                selected = [
                    row for row in rows
                    if row["formula"] == formula
                    and row["training_method"] == method
                    and row["checkpoint_epoch"] == epoch
                ]
                payload["by_formula"][formula][method][str(epoch)] = aggregate(selected)

    ordered = lambda method, epoch: sorted(
        (
            row for row in rows
            if row["training_method"] == method
            and row["checkpoint_epoch"] == epoch
        ),
        key=lambda row: (row["formula"], row["training_seed"]),
    )
    metrics = INT_METRICS[1:] + FLOAT_METRICS
    for method in METHODS:
        payload["paired"][f"{method}_epoch25_minus_base"] = {}
        before = ordered(method, 0)
        after = ordered(method, 25)
        for metric in metrics:
            payload["paired"][f"{method}_epoch25_minus_base"][metric] = paired_test(
                np.asarray([row[metric] for row in after], dtype=float),
                np.asarray([row[metric] for row in before], dtype=float),
            )

    peak = ordered("peak", 25)
    penalized = ordered("peak_penalized", 25)
    payload["paired"]["epoch25_penalized_minus_peak"] = {}
    for metric in metrics:
        payload["paired"]["epoch25_penalized_minus_peak"][metric] = paired_test(
            np.asarray([row[metric] for row in penalized], dtype=float),
            np.asarray([row[metric] for row in peak], dtype=float),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
