#!/usr/bin/env python
"""Audit and summarize the controlled cosine-versus-peak PPO matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


def _read_data(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=" ", skipinitialspace=True))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _slope(values: list[float]) -> float:
    x = np.arange(1, len(values) + 1, dtype=float)
    return float(np.polyfit(x, np.asarray(values), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    manifest = json.loads((root / "training_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"training manifest is not complete: {manifest.get('status')}")
    if len(manifest.get("runs", [])) != 24:
        raise RuntimeError(f"expected 24 runs, got {len(manifest.get('runs', []))}")

    epoch_rows: list[dict] = []
    run_rows: list[dict] = []
    for record in manifest["runs"]:
        if record.get("status") not in {"complete", "skipped_complete"}:
            raise RuntimeError(f"incomplete run in manifest: {record}")
        run_dir = Path(record["run_dir"])
        config = json.loads((run_dir / "run_config.json").read_text())
        data = _read_data(run_dir / "data.txt")
        if len(data) != args.epochs:
            raise RuntimeError(f"{record['run_id']}: expected {args.epochs} data rows")
        score_files = sorted(run_dir.glob("xrd_scores_*.csv"))
        checkpoints = sorted(run_dir.glob("epoch_*.pkl"))
        candidate_dirs = sorted(run_dir.glob("xrd_scores_*_candidates"))
        if len(score_files) != args.epochs or len(checkpoints) != args.epochs // 5:
            raise RuntimeError(f"{record['run_id']}: missing scores or checkpoints")

        all_scores = []
        candidate_cifs = 0
        for index, (row, score_file) in enumerate(zip(data, score_files), 1):
            with score_file.open() as handle:
                scores = [float(item["xrd_similarity"]) for item in csv.DictReader(handle)]
            if len(scores) != 8:
                raise RuntimeError(f"{record['run_id']}: {score_file.name} does not contain 8 scores")
            all_scores.extend(scores)
            candidate_dir = run_dir / f"{score_file.stem}_candidates"
            epoch_cifs = len(list(candidate_dir.glob("*.cif")))
            candidate_cifs += epoch_cifs
            epoch_rows.append({
                "run_id": record["run_id"],
                "formula": record["formula"],
                "method": record["method"],
                "seed": record["seed"],
                "epoch": index,
                "raw_score_mean": float(row["xrd_mean"]),
                "raw_score_sem": float(row["xrd_err"]),
                "raw_score_max": float(row["xrd_max"]),
                "raw_score_min": float(row["xrd_min"]),
                "zero_scores": sum(score == 0.0 for score in scores),
                "cifs_written": epoch_cifs,
                "attempts": int(row["attempt"]),
                "unique_space_groups": int(row["unique_space_groups"]),
                "unique_wyckoff_sequences": int(row["unique_wyckoff_sequences"]),
                "unique_atom_sequences": int(row["unique_atom_sequences"]),
                "unique_WA_combinations": int(row["unique_WA_combinations"]),
                "reconstructed_mean_ppo_reward": mean(scores) - max(all_scores),
                "reported_kl_log_ratio": float(row["kl"]),
                "entropy_g": float(row["g"]),
                "entropy_w": float(row["w"]),
                "entropy_a": float(row["a"]),
                "entropy_xyz": float(row["xyz"]),
                "entropy_l": float(row["l"]),
            })

        values = [float(row["xrd_mean"]) for row in data]
        guard = json.loads((Path(record["guard_dir"]) / "guard_summary.json").read_text())
        run_rows.append({
            "run_id": record["run_id"],
            "formula": record["formula"],
            "method": record["method"],
            "seed": record["seed"],
            "epochs": len(data),
            "training_candidates": len(all_scores),
            "cifs_written": candidate_cifs,
            "zero_scores": sum(score == 0.0 for score in all_scores),
            "mean_score_all_epochs": mean(values),
            "first_5_epoch_mean": mean(values[:5]),
            "last_5_epoch_mean": mean(values[-5:]),
            "last_minus_first_5": mean(values[-5:]) - mean(values[:5]),
            "linear_slope_per_epoch": _slope(values),
            "best_epoch_by_batch_mean": int(np.argmax(values)) + 1,
            "best_epoch_mean": max(values),
            "final_epoch_mean": values[-1],
            "mean_attempts": mean(int(row["attempt"]) for row in data),
            "mean_unique_WA": mean(int(row["unique_WA_combinations"]) for row in data),
            "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
            "min_available_gib": guard.get("min_available_gib_seen", ""),
            "min_swap_free_gib": guard.get("min_swap_free_gib_seen", ""),
            "guard_return_code": guard.get("return_code", ""),
            "learning_rate": config["lr"],
            "ppo_epochs": config["ppo_epochs"],
            "batch_size": config["batchsize"],
            "microbatch_size": config["ppo_microbatch_size"],
            "eps_clip": config["eps_clip"],
            "beta": config["beta"],
            "gamma": config["gamma"],
        })

    if len(epoch_rows) != 480:
        raise RuntimeError(f"expected 480 epoch rows, got {len(epoch_rows)}")

    group_rows = []
    for formula in sorted({row["formula"] for row in run_rows}):
        for method in ("cosine", "peak"):
            members = [row for row in run_rows if row["formula"] == formula and row["method"] == method]
            changes = [row["last_minus_first_5"] for row in members]
            slopes = [row["linear_slope_per_epoch"] for row in members]
            group_rows.append({
                "formula": formula,
                "method": method,
                "runs": len(members),
                "training_candidates": sum(row["training_candidates"] for row in members),
                "zero_scores": sum(row["zero_scores"] for row in members),
                "last_minus_first_5_mean": mean(changes),
                "last_minus_first_5_std": pstdev(changes),
                "positive_change_runs": sum(change > 0 for change in changes),
                "slope_mean": mean(slopes),
                "final_epoch_mean_across_seeds": mean(row["final_epoch_mean"] for row in members),
                "mean_unique_WA": mean(row["mean_unique_WA"] for row in members),
            })

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "PPO_TRAINING_EPOCH_METRICS.csv", epoch_rows)
    _write_csv(output / "PPO_TRAINING_RUN_SUMMARY.csv", run_rows)
    _write_csv(output / "PPO_TRAINING_GROUP_SUMMARY.csv", group_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    formulas = ["Zr", "NbS2", "Li2TeC2", "NaNiH3"]
    for column, formula in enumerate(formulas):
        for row_index, method in enumerate(("cosine", "peak")):
            axis = axes[row_index, column]
            selected = [row for row in epoch_rows if row["formula"] == formula and row["method"] == method]
            by_seed = {}
            for row in selected:
                by_seed.setdefault(row["seed"], []).append(row)
            matrix = []
            for seed, rows in sorted(by_seed.items()):
                rows.sort(key=lambda item: item["epoch"])
                values = [item["raw_score_mean"] for item in rows]
                matrix.append(values)
                axis.plot(range(1, args.epochs + 1), values, alpha=0.28, linewidth=0.9)
            array = np.asarray(matrix)
            mean_curve = array.mean(axis=0)
            std_curve = array.std(axis=0)
            axis.plot(range(1, args.epochs + 1), mean_curve, color="black", linewidth=1.8, label="3-seed mean")
            axis.fill_between(range(1, args.epochs + 1), mean_curve - std_curve, mean_curve + std_curve, color="black", alpha=0.12)
            axis.set_title(f"{formula} - {method}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("PPO epoch")
            if column == 0:
                axis.set_ylabel("Raw training score")
    figure.suptitle("Raw training scores (methods use different score scales)")
    figure.tight_layout()
    figure.savefig(output / "PPO_TRAINING_SCORE_CURVES.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for column, formula in enumerate(formulas):
        for row_index, method in enumerate(("cosine", "peak")):
            axis = axes[row_index, column]
            selected = [row for row in epoch_rows if row["formula"] == formula and row["method"] == method]
            by_seed = {}
            for row in selected:
                by_seed.setdefault(row["seed"], []).append(row)
            matrix = []
            for seed, rows in sorted(by_seed.items()):
                rows.sort(key=lambda item: item["epoch"])
                values = [item["reconstructed_mean_ppo_reward"] for item in rows]
                matrix.append(values)
                axis.plot(range(1, args.epochs + 1), values, alpha=0.28, linewidth=0.9)
            array = np.asarray(matrix)
            mean_curve = array.mean(axis=0)
            std_curve = array.std(axis=0)
            axis.plot(range(1, args.epochs + 1), mean_curve, color="black", linewidth=1.8)
            axis.fill_between(range(1, args.epochs + 1), mean_curve - std_curve, mean_curve + std_curve, color="black", alpha=0.12)
            axis.axhline(0.0, color="gray", linewidth=0.7)
            axis.set_title(f"{formula} - {method}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("PPO epoch")
            if column == 0:
                axis.set_ylabel("Mean centered PPO reward")
    figure.suptitle("Reconstructed PPO reward: score minus best score seen so far")
    figure.tight_layout()
    figure.savefig(output / "PPO_TRAINING_REWARD_CURVES.png", dpi=180)
    plt.close(figure)

    audit = {
        "manifest_status": manifest["status"],
        "runs": len(run_rows),
        "epoch_rows": len(epoch_rows),
        "training_candidates": sum(row["training_candidates"] for row in run_rows),
        "cifs_written": sum(row["cifs_written"] for row in run_rows),
        "zero_scores": sum(row["zero_scores"] for row in run_rows),
        "checkpoints": len(list(root.glob("runs/**/epoch_*.pkl"))),
        "score_files": len(list(root.glob("runs/**/xrd_scores_*.csv"))),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "min_available_gib": min(float(row["min_available_gib"]) for row in run_rows),
        "min_swap_free_gib": min(float(row["min_swap_free_gib"]) for row in run_rows),
    }
    (output / "PPO_TRAINING_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
