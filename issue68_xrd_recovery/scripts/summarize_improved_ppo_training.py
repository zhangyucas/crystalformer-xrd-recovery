#!/usr/bin/env python
"""Export and plot the complete improved XRD-PPO training trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np


FORMULAS = ("Zr", "NbS2", "Li2TeC2", "NaNiH3")
METHODS = ("peak", "peak_penalized")
FLOAT_COLUMNS = (
    "xrd_mean", "xrd_err", "xrd_max", "xrd_min",
    "reward_mean", "reward_err", "reward_max", "reward_min",
    "advantage_mean", "advantage_std", "ppo_objective",
    "log_ratio_to_reference", "clip_fraction", "ratio_mean", "ratio_max",
    "approx_kl_old", "grad_norm", "score_p10", "score_p50", "score_p90",
)
INT_COLUMNS = (
    "attempt", "unique_space_groups", "unique_wyckoff_sequences",
    "unique_atom_sequences", "unique_WA_combinations", "replay_epoch",
)


def read_training_rows(root: Path) -> list[dict]:
    manifest = json.loads((root / "training_manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 24:
        raise RuntimeError("expected a complete 24-run training manifest")

    rows = []
    for record in manifest["runs"]:
        with (Path(record["run_dir"]) / "data.txt").open() as handle:
            run_rows = list(csv.DictReader(handle, delimiter=" ", skipinitialspace=True))
        if len(run_rows) != manifest["epochs"]:
            raise RuntimeError(f"{record['run_id']}: incomplete epoch data")
        for epoch, row in enumerate(run_rows, 1):
            converted = {
                "run_id": record["run_id"],
                "formula": record["formula"],
                "method": record["method"],
                "seed": int(record["seed"]),
                "epoch": epoch,
            }
            converted.update({name: float(row[name]) for name in FLOAT_COLUMNS})
            converted.update({name: int(row[name]) for name in INT_COLUMNS})
            rows.append(converted)
    if len(rows) != 600:
        raise RuntimeError(f"expected 600 epoch rows, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for formula in FORMULAS:
        for method in METHODS:
            selected = [
                row for row in rows
                if row["formula"] == formula and row["method"] == method
            ]
            first = [row for row in selected if row["epoch"] <= 5]
            last = [row for row in selected if row["epoch"] >= 21]
            output.append({
                "formula": formula,
                "method": method,
                "seed_count": len({row["seed"] for row in selected}),
                "epoch_seed_points": len(selected),
                "objective_first5": mean(row["ppo_objective"] for row in first),
                "objective_last5": mean(row["ppo_objective"] for row in last),
                "objective_last5_minus_first5": (
                    mean(row["ppo_objective"] for row in last)
                    - mean(row["ppo_objective"] for row in first)
                ),
                "raw_score_first5": mean(row["xrd_mean"] for row in first),
                "raw_score_last5": mean(row["xrd_mean"] for row in last),
                "raw_score_last5_minus_first5": (
                    mean(row["xrd_mean"] for row in last)
                    - mean(row["xrd_mean"] for row in first)
                ),
                "clip_fraction_mean": mean(row["clip_fraction"] for row in selected),
                "approx_kl_old_mean": mean(row["approx_kl_old"] for row in selected),
                "ratio_mean": mean(row["ratio_mean"] for row in selected),
            })
    return output


def plot_metric(rows: list[dict], output: Path, metric: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for column, formula in enumerate(FORMULAS):
        for row_index, method in enumerate(METHODS):
            axis = axes[row_index, column]
            selected = [
                row for row in rows
                if row["formula"] == formula and row["method"] == method
            ]
            by_seed = {
                seed: sorted(
                    (row for row in selected if row["seed"] == seed),
                    key=lambda row: row["epoch"],
                )
                for seed in sorted({row["seed"] for row in selected})
            }
            matrix = []
            for seed, seed_rows in by_seed.items():
                values = [row[metric] for row in seed_rows]
                matrix.append(values)
                axis.plot(range(1, 26), values, alpha=0.45, linewidth=0.9, label=str(seed))
            values = np.asarray(matrix)
            average = values.mean(axis=0)
            spread = values.std(axis=0)
            axis.plot(range(1, 26), average, color="black", linewidth=1.7, label="mean")
            axis.fill_between(
                range(1, 26), average - spread, average + spread,
                color="black", alpha=0.12,
            )
            axis.axhline(0, color="gray", linewidth=0.6, alpha=0.6)
            axis.set_title(f"{formula} - {method}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("Outer epoch")
            if column == 0:
                axis.set_ylabel(ylabel)
            axis.legend(fontsize=7, ncol=2)
    figure.suptitle(f"Improved XRD-PPO: {ylabel} (3 seeds per panel)")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_training_rows(Path(args.root).resolve())
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    write_csv(output / "IMPROVED_PPO_EPOCH_METRICS.csv", rows)
    write_csv(output / "IMPROVED_PPO_FIRST_LAST_SUMMARY.csv", summary)
    plot_metric(rows, output / "IMPROVED_PPO_OBJECTIVE_ALL_RUNS.png", "ppo_objective", "PPO objective")
    plot_metric(rows, output / "IMPROVED_PPO_RAW_XRD_ALL_RUNS.png", "xrd_mean", "Training-batch XRD score")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
