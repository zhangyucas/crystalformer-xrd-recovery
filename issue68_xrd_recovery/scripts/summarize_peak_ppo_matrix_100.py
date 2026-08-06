#!/usr/bin/env python
"""Audit and plot the complete detailed 100-epoch PPO matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


FORMULAS = ("Zr", "NbS2", "Li2TeC2", "NaNiH3")
METHODS = ("cosine", "peak")


def read_data(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=" ", skipinitialspace=True))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linear_slope(values: list[float]) -> float:
    return float(np.polyfit(np.arange(1, len(values) + 1), values, 1)[0])


def paired_bootstrap_ci(
    differences: list[float], *, seed: int, resamples: int = 20_000
) -> tuple[float, float]:
    """Return a deterministic seed-level bootstrap CI for a mean difference."""
    random = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    draws = random.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    manifest = json.loads((root / "training_manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 24:
        raise RuntimeError("the 24-run training manifest is not complete")

    epoch_rows = []
    run_rows = []
    for record in manifest["runs"]:
        run_dir = Path(record["run_dir"])
        data = read_data(run_dir / "data.txt")
        if len(data) != args.epochs:
            raise RuntimeError(f"{record['run_id']}: expected {args.epochs} epochs")
        required = {
            "reward_mean", "reward_err", "reward_max", "reward_min",
            "advantage_mean", "advantage_std", "ppo_objective", "log_ratio_to_reference",
        }
        if not required.issubset(data[0]):
            raise RuntimeError(f"{record['run_id']}: detailed PPO columns are missing")
        score_files = sorted(run_dir.glob("xrd_scores_*.csv"))
        checkpoints = sorted(run_dir.glob("epoch_*.pkl"))
        if len(score_files) != args.epochs or len(checkpoints) != args.epochs // 10:
            raise RuntimeError(f"{record['run_id']}: missing scores or checkpoints")

        all_scores = []
        cifs = 0
        for index, (row, score_file) in enumerate(zip(data, score_files), 1):
            with score_file.open() as handle:
                scores = [float(item["xrd_similarity"]) for item in csv.DictReader(handle)]
            if len(scores) != 8:
                raise RuntimeError(f"{score_file}: expected 8 scores")
            all_scores.extend(scores)
            epoch_cifs = len(list((run_dir / f"{score_file.stem}_candidates").glob("*.cif")))
            cifs += epoch_cifs
            epoch_rows.append({
                "run_id": record["run_id"], "formula": record["formula"],
                "method": record["method"], "seed": record["seed"], "epoch": index,
                "raw_score_mean": float(row["xrd_mean"]),
                "raw_score_sem": float(row["xrd_err"]),
                "raw_score_max": float(row["xrd_max"]),
                "raw_score_min": float(row["xrd_min"]),
                "reward_mean": float(row["reward_mean"]),
                "reward_sem": float(row["reward_err"]),
                "reward_max": float(row["reward_max"]),
                "reward_min": float(row["reward_min"]),
                "advantage_mean": float(row["advantage_mean"]),
                "advantage_std": float(row["advantage_std"]),
                "ppo_objective": float(row["ppo_objective"]),
                "log_ratio_to_reference": float(row["log_ratio_to_reference"]),
                "attempts": int(row["attempt"]),
                "unique_space_groups": int(row["unique_space_groups"]),
                "unique_wyckoff_sequences": int(row["unique_wyckoff_sequences"]),
                "unique_atom_sequences": int(row["unique_atom_sequences"]),
                "unique_WA_combinations": int(row["unique_WA_combinations"]),
                "negative_logp_g": float(row["g"]), "negative_logp_w": float(row["w"]),
                "negative_logp_a": float(row["a"]), "negative_logp_xyz": float(row["xyz"]),
                "negative_logp_l": float(row["l"]),
                "zero_scores": sum(score == 0.0 for score in scores),
                "cifs_written": epoch_cifs,
            })

        raw = [float(row["xrd_mean"]) for row in data]
        rewards = [float(row["reward_mean"]) for row in data]
        objectives = [float(row["ppo_objective"]) for row in data]
        guard = json.loads((Path(record["guard_dir"]) / "guard_summary.json").read_text())
        run_rows.append({
            "run_id": record["run_id"], "formula": record["formula"],
            "method": record["method"], "seed": record["seed"],
            "epochs": args.epochs, "training_candidates": len(all_scores),
            "cifs_written": cifs, "zero_scores": sum(score == 0.0 for score in all_scores),
            "raw_score_first20": mean(raw[:20]), "raw_score_last20": mean(raw[-20:]),
            "raw_score_last_minus_first20": mean(raw[-20:]) - mean(raw[:20]),
            "raw_score_slope": linear_slope(raw),
            "reward_first20": mean(rewards[:20]), "reward_last20": mean(rewards[-20:]),
            "reward_last_minus_first20": mean(rewards[-20:]) - mean(rewards[:20]),
            "reward_slope": linear_slope(rewards),
            "objective_first20": mean(objectives[:20]), "objective_last20": mean(objectives[-20:]),
            "objective_last_minus_first20": mean(objectives[-20:]) - mean(objectives[:20]),
            "objective_slope": linear_slope(objectives),
            "best_raw_epoch": int(np.argmax(raw)) + 1, "best_raw_score": max(raw),
            "best_reward_epoch": int(np.argmax(rewards)) + 1, "best_reward_mean": max(rewards),
            "mean_attempts": mean(int(row["attempt"]) for row in data),
            "mean_unique_WA": mean(int(row["unique_WA_combinations"]) for row in data),
            "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
            "min_available_gib": guard.get("min_available_gib_seen", ""),
            "min_swap_free_gib": guard.get("min_swap_free_gib_seen", ""),
        })

    if len(epoch_rows) != 24 * args.epochs:
        raise RuntimeError(f"expected {24 * args.epochs} epoch rows")
    write_csv(output / "PPO_100_EPOCH_METRICS.csv", epoch_rows)
    write_csv(output / "PPO_100_RUN_SUMMARY.csv", run_rows)

    change_rows = []
    for formula_index, formula in enumerate(FORMULAS):
        for method_index, method in enumerate(METHODS):
            for metric_index, metric in enumerate(
                ("raw_score_mean", "reward_mean", "ppo_objective")
            ):
                changes = []
                for seed in sorted({
                    row["seed"] for row in epoch_rows
                    if row["formula"] == formula and row["method"] == method
                }):
                    first = [
                        row[metric] for row in epoch_rows
                        if row["formula"] == formula and row["method"] == method
                        and row["seed"] == seed and 1 <= row["epoch"] <= 20
                    ]
                    last = [
                        row[metric] for row in epoch_rows
                        if row["formula"] == formula and row["method"] == method
                        and row["seed"] == seed and 81 <= row["epoch"] <= 100
                    ]
                    changes.append(mean(last) - mean(first))
                low, high = paired_bootstrap_ci(
                    changes,
                    seed=(
                        10_000 + formula_index * 100 + method_index * 10 + metric_index
                    ),
                )
                change_rows.append({
                    "formula": formula,
                    "method": method,
                    "metric": metric,
                    "seed_pairs": len(changes),
                    "last20_minus_first20_mean": mean(changes),
                    "last20_minus_first20_std": pstdev(changes),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "positive_seed_pairs": sum(value > 0 for value in changes),
                    "negative_seed_pairs": sum(value < 0 for value in changes),
                    "zero_seed_pairs": sum(value == 0 for value in changes),
                })
    write_csv(output / "PPO_100_WITHIN_METHOD_CHANGES.csv", change_rows)

    segment_rows = []
    for formula in FORMULAS:
        for method in METHODS:
            members = [row for row in epoch_rows if row["formula"] == formula and row["method"] == method]
            for start in range(1, args.epochs + 1, 10):
                selected = [row for row in members if start <= row["epoch"] <= start + 9]
                segment_rows.append({
                    "formula": formula, "method": method, "epoch_start": start,
                    "epoch_end": start + 9, "epoch_seed_points": len(selected),
                    "raw_score_mean": mean(row["raw_score_mean"] for row in selected),
                    "raw_score_std": pstdev(row["raw_score_mean"] for row in selected),
                    "reward_mean": mean(row["reward_mean"] for row in selected),
                    "reward_std": pstdev(row["reward_mean"] for row in selected),
                    "ppo_objective_mean": mean(row["ppo_objective"] for row in selected),
                    "ppo_objective_std": pstdev(row["ppo_objective"] for row in selected),
                    "zero_scores": sum(row["zero_scores"] for row in selected),
                    "mean_unique_WA": mean(row["unique_WA_combinations"] for row in selected),
                })
    write_csv(output / "PPO_100_SEGMENT_SUMMARY.csv", segment_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"cosine": "#4472c4", "peak": "#e07a1f"}
    for formula in FORMULAS:
        figure, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
        for column, method in enumerate(METHODS):
            selected = [row for row in epoch_rows if row["formula"] == formula and row["method"] == method]
            by_seed = {}
            for row in selected:
                by_seed.setdefault(row["seed"], []).append(row)
            for seed, rows in sorted(by_seed.items()):
                rows.sort(key=lambda item: item["epoch"])
                x = [row["epoch"] for row in rows]
                axes[0, column].plot(x, [row["reward_mean"] for row in rows], linewidth=1.0, label=f"seed {seed}")
                axes[1, column].plot(x, [row["raw_score_mean"] for row in rows], linewidth=1.0, label=f"seed {seed}")
                axes[2, column].plot(x, [row["ppo_objective"] for row in rows], linewidth=1.0, label=f"seed {seed}")
            axes[0, column].set_title(f"{formula} - {method}")
            axes[0, column].set_ylabel("Mean PPO reward")
            axes[1, column].set_ylabel("Raw XRD score")
            axes[2, column].set_ylabel("PPO objective")
            axes[2, column].set_xlabel("Epoch")
            for axis in axes[:, column]:
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8, ncol=3)
        figure.suptitle(f"{formula}: complete 100-epoch trajectories")
        figure.tight_layout()
        figure.savefig(output / f"PPO_100_COMPLETE_{formula}.png", dpi=200)
        plt.close(figure)

    for metric, filename, ylabel in (
        ("reward_mean", "PPO_100_REWARD_ALL_RUNS.png", "Mean PPO reward"),
        ("raw_score_mean", "PPO_100_RAW_SCORE_ALL_RUNS.png", "Raw XRD score"),
        ("ppo_objective", "PPO_100_OBJECTIVE_ALL_RUNS.png", "PPO objective"),
    ):
        figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
        for column, formula in enumerate(FORMULAS):
            for row_index, method in enumerate(METHODS):
                axis = axes[row_index, column]
                selected = [row for row in epoch_rows if row["formula"] == formula and row["method"] == method]
                by_seed = {}
                for row in selected:
                    by_seed.setdefault(row["seed"], []).append(row)
                matrix = []
                for seed, rows in sorted(by_seed.items()):
                    rows.sort(key=lambda item: item["epoch"])
                    values = [row[metric] for row in rows]
                    matrix.append(values)
                    axis.plot(range(1, args.epochs + 1), values, alpha=0.35, linewidth=0.75)
                array = np.asarray(matrix)
                average = array.mean(axis=0)
                spread = array.std(axis=0)
                axis.plot(range(1, args.epochs + 1), average, color="black", linewidth=1.5)
                axis.fill_between(range(1, args.epochs + 1), average - spread, average + spread, color="black", alpha=0.12)
                axis.set_title(f"{formula} - {method}")
                axis.grid(alpha=0.2)
                if row_index == 1:
                    axis.set_xlabel("Epoch")
                if column == 0:
                    axis.set_ylabel(ylabel)
        figure.suptitle(f"Complete 100-epoch {ylabel.lower()} trajectories")
        figure.tight_layout()
        figure.savefig(output / filename, dpi=200)
        plt.close(figure)

    audit = {
        "status": manifest["status"], "runs": len(run_rows), "epochs_per_run": args.epochs,
        "epoch_rows": len(epoch_rows),
        "training_candidates": sum(row["training_candidates"] for row in run_rows),
        "cifs_written": sum(row["cifs_written"] for row in run_rows),
        "zero_scores": sum(row["zero_scores"] for row in run_rows),
        "checkpoints": len(list(root.glob("runs/**/epoch_*.pkl"))),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "min_available_gib": min(float(row["min_available_gib"]) for row in run_rows),
        "min_swap_free_gib": min(float(row["min_swap_free_gib"]) for row in run_rows),
    }
    (output / "PPO_100_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
