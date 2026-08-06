#!/usr/bin/env python
"""Summarize paired pre-PPO, cosine-PPO, and peak-PPO evaluations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


METRICS = (
    "mean_peak_score",
    "mean_cosine_score",
    "structure_matches",
    "valid_xrd",
    "formula_match_sampler",
    "top10_peak_structure_matches",
    "top10_cosine_structure_matches",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_flip_pvalue(differences: list[float]) -> float:
    """Exact two-sided paired randomization p-value for three seeds."""
    observed = abs(mean(differences))
    if observed == 0:
        return 1.0
    values = []
    for mask in range(1 << len(differences)):
        signed = [value if mask & (1 << index) else -value for index, value in enumerate(differences)]
        values.append(abs(mean(signed)))
    return sum(value >= observed - 1e-15 for value in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    trained = _read(Path(args.trained))
    base = _read(Path(args.base))
    trained_by_key = {
        (row["formula"], int(row["test_seed"]), row["training_method"]): row
        for row in trained
    }
    base_by_key = {(row["formula"], int(row["test_seed"])): row for row in base}
    formulas = ["Zr", "NbS2", "Li2TeC2", "NaNiH3"]

    paired_rows = []
    for formula in formulas:
        test_seeds = sorted(seed for f, seed in base_by_key if f == formula)
        for seed in test_seeds:
            base_row = base_by_key[(formula, seed)]
            cosine = trained_by_key[(formula, seed, "cosine")]
            peak = trained_by_key[(formula, seed, "peak")]
            row = {"formula": formula, "test_seed": seed}
            for metric in METRICS:
                base_value = float(base_row[metric])
                cosine_value = float(cosine[metric])
                peak_value = float(peak[metric])
                row[f"base_{metric}"] = base_value
                row[f"cosine_{metric}"] = cosine_value
                row[f"peak_{metric}"] = peak_value
                row[f"peak_minus_cosine_{metric}"] = peak_value - cosine_value
                row[f"cosine_minus_base_{metric}"] = cosine_value - base_value
                row[f"peak_minus_base_{metric}"] = peak_value - base_value
            paired_rows.append(row)
    _write(output / "PPO_INDEPENDENT_PAIRED_RESULTS.csv", paired_rows)

    group_rows = []
    comparisons = ("peak_minus_cosine", "cosine_minus_base", "peak_minus_base")
    for formula in formulas + ["ALL"]:
        members = paired_rows if formula == "ALL" else [row for row in paired_rows if row["formula"] == formula]
        for metric in METRICS:
            for comparison in comparisons:
                differences = [row[f"{comparison}_{metric}"] for row in members]
                group_rows.append({
                    "formula": formula,
                    "metric": metric,
                    "comparison": comparison,
                    "pairs": len(differences),
                    "mean_difference": mean(differences),
                    "std_difference": pstdev(differences),
                    "positive_pairs": sum(value > 0 for value in differences),
                    "negative_pairs": sum(value < 0 for value in differences),
                    "zero_pairs": sum(value == 0 for value in differences),
                    "exact_sign_flip_p_two_sided": _sign_flip_pvalue(differences),
                })
    _write(output / "PPO_INDEPENDENT_COMPARISONS.csv", group_rows)

    model_rows = []
    for formula in formulas + ["ALL"]:
        trained_members = trained if formula == "ALL" else [row for row in trained if row["formula"] == formula]
        base_members = base if formula == "ALL" else [row for row in base if row["formula"] == formula]
        for method, members in (("base", base_members), ("cosine", [row for row in trained_members if row["training_method"] == "cosine"]), ("peak", [row for row in trained_members if row["training_method"] == "peak"])):
            model_rows.append({
                "formula": formula,
                "method": method,
                "runs": len(members),
                "samples": sum(int(row["samples"]) for row in members),
                "formula_match_sampler": sum(int(row["formula_match_sampler"]) for row in members),
                "valid_xrd": sum(int(row["valid_xrd"]) for row in members),
                "structure_matches": sum(int(row["structure_matches"]) for row in members),
                "mean_peak_score": mean(float(row["mean_peak_score"]) for row in members),
                "std_peak_score_across_runs": pstdev(float(row["mean_peak_score"]) for row in members),
                "mean_cosine_score": mean(float(row["mean_cosine_score"]) for row in members),
                "std_cosine_score_across_runs": pstdev(float(row["mean_cosine_score"]) for row in members),
                "top10_peak_structure_matches": sum(int(row["top10_peak_structure_matches"]) for row in members),
                "top10_cosine_structure_matches": sum(int(row["top10_cosine_structure_matches"]) for row in members),
            })
    _write(output / "PPO_INDEPENDENT_MODEL_SUMMARY.csv", model_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    methods = ("base", "cosine", "peak")
    colors = ("#777777", "#4472c4", "#e07a1f")
    for axis, metric, ylabel in zip(
        axes,
        ("mean_peak_score", "mean_cosine_score", "structure_matches"),
        ("Common peak score", "Common cosine score", "StructureMatcher hits / 100"),
    ):
        x = np.arange(len(formulas))
        width = 0.24
        for method_index, (method, color) in enumerate(zip(methods, colors)):
            values = []
            errors = []
            for formula in formulas:
                if method == "base":
                    members = [base_by_key[(formula, seed)] for seed in sorted(seed for f, seed in base_by_key if f == formula)]
                else:
                    members = [trained_by_key[(formula, seed, method)] for seed in sorted(seed for f, seed in base_by_key if f == formula)]
                metric_values = [float(row[metric]) for row in members]
                values.append(mean(metric_values))
                errors.append(pstdev(metric_values))
            axis.bar(x + (method_index - 1) * width, values, width, yerr=errors, capsize=2, label=method, color=color)
        axis.set_xticks(x, formulas)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    figure.suptitle("Independent evaluation, 3 paired seeds per material")
    figure.tight_layout()
    figure.savefig(output / "PPO_INDEPENDENT_COMPARISON.png", dpi=180)


if __name__ == "__main__":
    main()
