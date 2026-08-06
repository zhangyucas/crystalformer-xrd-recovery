#!/usr/bin/env python
"""Summarize paired unseen-sample evaluations across PPO checkpoints."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


FORMULAS = ("Zr", "NbS2", "Li2TeC2", "NaNiH3")
METHODS = ("base", "cosine", "peak")
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


def _exact_sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(mean(differences))
    null = [
        abs(mean(sign * value for sign, value in zip(signs, differences)))
        for signs in product((-1, 1), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-15 for value in null) / len(null)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = _read(Path(args.checkpoints))
    base_rows = _read(Path(args.base))
    epochs = sorted({int(row["checkpoint_epoch"]) for row in checkpoint_rows})
    checkpoint_by_key = {
        (
            row["formula"], int(row["test_seed"]), row["training_method"],
            int(row["checkpoint_epoch"]),
        ): row
        for row in checkpoint_rows
    }
    base_by_key = {
        (row["formula"], int(row["test_seed"])): row for row in base_rows
    }
    expected = len(FORMULAS) * 3 * 2 * len(epochs)
    if len(checkpoint_rows) != expected or len(base_rows) != len(FORMULAS) * 3:
        raise RuntimeError("checkpoint or base evaluation matrix is incomplete")

    model_rows = []
    for formula in FORMULAS + ("ALL",):
        formulas = FORMULAS if formula == "ALL" else (formula,)
        for epoch in (0, *epochs):
            methods = ("base",) if epoch == 0 else ("cosine", "peak")
            for method in methods:
                if method == "base":
                    members = [
                        row for (member_formula, _), row in base_by_key.items()
                        if member_formula in formulas
                    ]
                else:
                    members = [
                        row for (member_formula, _, member_method, member_epoch), row
                        in checkpoint_by_key.items()
                        if member_formula in formulas
                        and member_method == method and member_epoch == epoch
                    ]
                model_rows.append({
                    "formula": formula,
                    "method": method,
                    "checkpoint_epoch": epoch,
                    "runs": len(members),
                    "samples": sum(int(row["samples"]) for row in members),
                    "formula_match_sampler": sum(
                        int(row["formula_match_sampler"]) for row in members
                    ),
                    "valid_xrd": sum(int(row["valid_xrd"]) for row in members),
                    "structure_matches": sum(
                        int(row["structure_matches"]) for row in members
                    ),
                    "mean_peak_score": mean(
                        float(row["mean_peak_score"]) for row in members
                    ),
                    "std_peak_score_across_runs": pstdev(
                        float(row["mean_peak_score"]) for row in members
                    ),
                    "mean_cosine_score": mean(
                        float(row["mean_cosine_score"]) for row in members
                    ),
                    "std_cosine_score_across_runs": pstdev(
                        float(row["mean_cosine_score"]) for row in members
                    ),
                    "top10_peak_structure_matches": sum(
                        int(row["top10_peak_structure_matches"]) for row in members
                    ),
                    "top10_cosine_structure_matches": sum(
                        int(row["top10_cosine_structure_matches"]) for row in members
                    ),
                })
    _write(output / "PPO_CHECKPOINT_MODEL_SUMMARY.csv", model_rows)

    comparisons = []
    for formula in FORMULAS + ("ALL",):
        formulas = FORMULAS if formula == "ALL" else (formula,)
        keys = sorted(key for key in base_by_key if key[0] in formulas)
        for epoch in epochs:
            for metric in METRICS:
                values = {
                    "peak_minus_cosine": [
                        float(checkpoint_by_key[(f, seed, "peak", epoch)][metric])
                        - float(checkpoint_by_key[(f, seed, "cosine", epoch)][metric])
                        for f, seed in keys
                    ],
                    "cosine_minus_base": [
                        float(checkpoint_by_key[(f, seed, "cosine", epoch)][metric])
                        - float(base_by_key[(f, seed)][metric])
                        for f, seed in keys
                    ],
                    "peak_minus_base": [
                        float(checkpoint_by_key[(f, seed, "peak", epoch)][metric])
                        - float(base_by_key[(f, seed)][metric])
                        for f, seed in keys
                    ],
                }
                for comparison, differences in values.items():
                    comparisons.append({
                        "formula": formula,
                        "checkpoint_epoch": epoch,
                        "metric": metric,
                        "comparison": comparison,
                        "pairs": len(differences),
                        "mean_difference": mean(differences),
                        "std_difference": pstdev(differences),
                        "positive_pairs": sum(value > 0 for value in differences),
                        "negative_pairs": sum(value < 0 for value in differences),
                        "zero_pairs": sum(value == 0 for value in differences),
                        "exact_sign_flip_p_two_sided": _exact_sign_flip_pvalue(
                            differences
                        ),
                    })
    _write(output / "PPO_CHECKPOINT_COMPARISONS.csv", comparisons)

    epoch_changes = []
    if 20 in epochs and 100 in epochs:
        for method in ("cosine", "peak"):
            for metric in METRICS:
                differences = [
                    float(checkpoint_by_key[(formula, seed, method, 100)][metric])
                    - float(checkpoint_by_key[(formula, seed, method, 20)][metric])
                    for formula, seed in sorted(base_by_key)
                ]
                epoch_changes.append({
                    "method": method,
                    "metric": metric,
                    "comparison": "epoch100_minus_epoch20",
                    "pairs": len(differences),
                    "mean_difference": mean(differences),
                    "std_difference": pstdev(differences),
                    "positive_pairs": sum(value > 0 for value in differences),
                    "negative_pairs": sum(value < 0 for value in differences),
                    "zero_pairs": sum(value == 0 for value in differences),
                    "exact_sign_flip_p_two_sided": _exact_sign_flip_pvalue(differences),
                })
        _write(output / "PPO_CHECKPOINT_EPOCH20_TO_100.csv", epoch_changes)

    audit = {
        "status": "complete",
        "checkpoint_runs": len(checkpoint_rows),
        "samples_per_checkpoint": sorted({int(row["samples"]) for row in checkpoint_rows}),
        "checkpoint_samples": sum(int(row["samples"]) for row in checkpoint_rows),
        "base_runs": len(base_rows),
        "base_samples": sum(int(row["samples"]) for row in base_rows),
        "checkpoint_epochs": epochs,
        "formula_matches_checkpoint": sum(
            int(row["formula_match_sampler"]) for row in checkpoint_rows
        ),
        "errors_checkpoint": sum(int(row["errors"]) for row in checkpoint_rows),
    }
    (output / "PPO_CHECKPOINT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metric_labels = (
        ("mean_peak_score", "Common peak score"),
        ("mean_cosine_score", "Common cosine score"),
        ("structure_matches", "StructureMatcher hits / 1200"),
    )
    colors = {"cosine": "#4472c4", "peak": "#e07a1f"}
    all_rows = [row for row in model_rows if row["formula"] == "ALL"]
    for axis, (metric, label) in zip(axes, metric_labels):
        base = next(row for row in all_rows if row["method"] == "base")
        axis.axhline(
            float(base[metric]), color="#666666", linestyle="--", label="pre-PPO"
        )
        for method in ("cosine", "peak"):
            rows = sorted(
                (row for row in all_rows if row["method"] == method),
                key=lambda row: int(row["checkpoint_epoch"]),
            )
            axis.plot(
                [int(row["checkpoint_epoch"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker="o", color=colors[method], label=method,
            )
        axis.set_xlabel("Checkpoint epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Independent paired evaluation across PPO checkpoints")
    figure.tight_layout()
    figure.savefig(output / "PPO_CHECKPOINT_COMPARISON.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
