#!/usr/bin/env python
"""Run the equal-budget peak versus scale-penalized XRD PPO matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time


CASES = (
    ("Zr", "case_02_Zr"),
    ("NbS2", "case_03_NbS2"),
    ("Li2TeC2", "case_07_Li2TeC2"),
    ("NaNiH3", "case_08_NaNiH3"),
)
METHODS = ("peak", "peak_penalized")
SEED_OFFSETS = (1, 2, 3)


def run_command(command: list[str], log_path: Path, timeout: float) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def find_run_dir(folder: Path) -> Path | None:
    matches = sorted(path.parent for path in folder.glob("*/data.txt"))
    return matches[0] if len(matches) == 1 else None


def completed_run(folder: Path, epochs: int) -> tuple[bool, str, Path | None]:
    run_dir = find_run_dir(folder)
    if run_dir is None:
        return False, "missing_or_ambiguous_run_dir", None
    with (run_dir / "data.txt").open() as handle:
        rows = list(csv.DictReader(handle, delimiter=" ", skipinitialspace=True))
    checkpoints = [
        run_dir / f"epoch_{10000 + epoch:06d}.pkl"
        for epoch in range(5, epochs + 1, 5)
    ]
    required = {
        "reward_mean", "advantage_std", "ppo_objective", "clip_fraction",
        "ratio_max", "approx_kl_old", "grad_norm", "score_p50",
    }
    if len(rows) != epochs:
        return False, f"expected_{epochs}_rows_got_{len(rows)}", run_dir
    if not rows or not required.issubset(rows[0]):
        return False, "missing_improved_ppo_metrics", run_dir
    if not all(path.is_file() and path.stat().st_size > 0 for path in checkpoints):
        return False, "missing_checkpoint", run_dir
    return True, "complete", run_dir


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-hours", type=float, default=14.5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--trainable-scope", choices=("all", "heads"), default="heads")
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--min-swap-free-gib", type=float, default=4.0)
    args = parser.parse_args()
    if (
        args.max_hours <= 0
        or args.epochs <= 0
        or args.epochs % 5
        or args.min_available_gib <= 0
        or args.min_swap_free_gib < 0
    ):
        parser.error(
            "max-hours and memory availability must be positive, epochs must be "
            "a multiple of 5, and swap availability must be non-negative"
        )

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    checkpoint = repo / "issue68_xrd_recovery/experiments/gpu_20260729/checkpoints/prior_params_epoch_010000.pkl"
    targets = repo / "issue68_xrd_recovery/experiments/benchmark_20260731/opxrd_subset"
    bridge = repo / "issue68_xrd_recovery/scripts/run_gpu_host.sh"
    guard = repo / "issue68_xrd_recovery/scripts/run_memory_guard.py"
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    manifest_path = root / "training_manifest.json"
    manifest = {
        "status": "running", "started_unix": time.time(), "max_hours": args.max_hours,
        "epochs": args.epochs, "batchsize": 32, "ppo_epochs": 4,
        "candidate_budget": args.epochs * 32,
        "optimizer_update_budget": args.epochs * 4,
        "trainable_scope": args.trainable_scope,
        "min_available_gib": args.min_available_gib,
        "min_swap_free_gib": args.min_swap_free_gib,
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    for case_index, (formula, case_dir) in enumerate(CASES, start=1):
        target = targets / case_dir / "target.csv"
        for seed_offset in SEED_OFFSETS:
            seed = 9700 + case_index * 10 + seed_offset
            for method in METHODS:
                run_id = f"{case_dir}_{method}_{args.trainable_scope}_seed{seed}"
                folder = root / "runs" / run_id
                complete, reason, run_dir = completed_run(folder, args.epochs) if folder.exists() else (False, "not_started", None)
                if complete:
                    manifest["runs"].append({"run_id": run_id, "status": "skipped_complete", "run_dir": str(run_dir)})
                    write_manifest(manifest_path, manifest)
                    continue
                remaining = deadline - time.monotonic()
                if remaining < 900:
                    manifest.update({"status": "deadline_stop", "stop_reason": "less_than_15_minutes"})
                    write_manifest(manifest_path, manifest)
                    return

                train = [
                    "conda", "run", "-n", "crystal_wsl", "env",
                    "XLA_PYTHON_CLIENT_PREALLOCATE=false", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.30",
                    "MALLOC_TRIM_THRESHOLD_=0", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                    "python", "-m", "crystalformer.cli.train_ppo",
                    "--reward", "xrd", "--xrd-score", method, "--formula", formula,
                    "--xrd_target", str(target), "--restore_path", str(checkpoint), "--folder", str(folder),
                    "--epochs", str(args.epochs), "--checkpoint-interval", "5", "--ppo_epochs", "4",
                    "--batchsize", "32", "--sampling-batchsize", "1", "--ppo-microbatch-size", "8",
                    "--sample_multiplier", "1", "--max_sampling_attempts", "256",
                    "--composition_max_atoms", "40", "--composition_size_bias", "0.5",
                    "--beta", "0.1", "--gamma", "0", "--alpha", "0", "--lr", "5e-7",
                    "--trainable-scope", args.trainable_scope,
                    "--reset_optimizer", "--seed", str(seed), "--K", "0", "--num_io_process", "1",
                ]
                command = [
                    str(bridge), "conda", "run", "-n", "crystal_wsl", "python", str(guard),
                    "--output-dir", str(root / "guards" / run_id),
                    "--min-available-gib", str(args.min_available_gib),
                    "--min-swap-free-gib", str(args.min_swap_free_gib),
                    "--interval", "0.5", "--", *train,
                ]
                log_path = root / "logs" / f"{run_id}.log"
                run_started = time.monotonic()
                return_code = run_command(command, log_path, remaining)
                complete, reason, run_dir = completed_run(folder, args.epochs) if folder.exists() else (False, "no_output", None)
                status = "complete" if return_code == 0 and complete else "failed"
                manifest["runs"].append({
                    "run_id": run_id, "formula": formula, "method": method,
                    "seed": seed, "status": status, "return_code": return_code,
                    "validation": reason, "elapsed_seconds": time.monotonic() - run_started,
                    "run_dir": None if run_dir is None else str(run_dir), "log": str(log_path),
                })
                write_manifest(manifest_path, manifest)
                if status != "complete":
                    manifest.update({"status": "failed", "stop_reason": run_id})
                    write_manifest(manifest_path, manifest)
                    return

    manifest.update({"status": "complete", "elapsed_seconds": time.monotonic() - started})
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
