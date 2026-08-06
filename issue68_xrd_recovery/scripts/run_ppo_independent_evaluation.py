#!/usr/bin/env python
"""Run bounded same-protocol sampling for the final controlled PPO models."""

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
    ("Zr", "case_02_Zr", 9410),
    ("NbS2", "case_03_NbS2", 9420),
    ("Li2TeC2", "case_07_Li2TeC2", 9430),
    ("NaNiH3", "case_08_NaNiH3", 9440),
)
METHODS = ("cosine", "peak")
SEED_OFFSETS = (1, 2, 3)


def _run(command: list[str], log_path: Path, timeout: float) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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


def _training_run(training_root: Path, case_dir: str, method: str, seed: int) -> Path:
    base = training_root / "runs" / f"{case_dir}_{method}_seed{seed}"
    matches = [path.parent for path in base.glob("*/data.txt")]
    if len(matches) != 1:
        raise RuntimeError(f"could not resolve training run: {base}")
    return matches[0]


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.max_hours <= 0:
        parser.error("sample count and time limit must be positive")

    repo = Path(__file__).resolve().parents[2]
    training_root = Path(args.training_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark = repo / "issue68_xrd_recovery/experiments/benchmark_20260731/opxrd_subset"
    bridge = repo / "issue68_xrd_recovery/scripts/run_gpu_host.sh"
    sampler = repo / "issue68_xrd_recovery/scripts/xrd_prior_sample.py"
    converter = repo / "issue68_xrd_recovery/scripts/xrd_recovery.py"
    evaluator = repo / "issue68_xrd_recovery/scripts/evaluate_ppo_sample_pool.py"
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    manifest_path = output_root / "evaluation_manifest.json"
    manifest = {"status": "running", "samples_per_model": args.num_samples, "runs": []}
    _write_manifest(manifest_path, manifest)

    for case_index, (formula, case_dir, test_seed_base) in enumerate(CASES, 1):
        target_dir = benchmark / case_dir
        for seed_offset in SEED_OFFSETS:
          training_seed = 9300 + case_index * 10 + seed_offset
          test_seed = test_seed_base + seed_offset
          for method in METHODS:
            run_id = f"{case_dir}_{method}_train{training_seed}_test{test_seed}"
            run_output = output_root / "runs" / run_id
            summary_path = run_output / "common_metrics.json"
            if summary_path.is_file():
                manifest["runs"].append({
                    "run_id": run_id, "formula": formula, "training_method": method,
                    "training_seed": training_seed, "test_seed": test_seed,
                    "status": "skipped_complete", "return_code": 0,
                    "elapsed_seconds": 0.0, "output_dir": str(run_output),
                })
                _write_manifest(manifest_path, manifest)
                continue
            remaining = deadline - time.monotonic()
            if remaining < 600:
                manifest.update({"status": "deadline_stop", "stop_reason": "less_than_10_minutes"})
                _write_manifest(manifest_path, manifest)
                return

            training_run = _training_run(training_root, case_dir, method, training_seed)
            checkpoint = training_run / "epoch_010020.pkl"
            params_checkpoint = run_output / "params_epoch_010020.pkl"
            samples_dir = run_output / "samples"
            cifs_dir = run_output / "candidates"
            commands = [
                ["conda", "run", "-n", "crystal_wsl", "python", str(sampler), "extract-params",
                 "--checkpoint", str(checkpoint), "--output", str(params_checkpoint)],
                [str(bridge), "conda", "run", "-n", "crystal_wsl", "env",
                 "XLA_PYTHON_CLIENT_PREALLOCATE=false", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.40",
                 "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                 "python", str(sampler), "sample", "--checkpoint", str(params_checkpoint),
                 "--output-dir", str(samples_dir), "--formula", formula,
                 "--num-samples", str(args.num_samples), "--batch-size", "2", "--seed", str(test_seed),
                 "--platform", "gpu", "--composition-max-atoms", "40",
                 "--composition-size-bias", "0.5", "--min-available-gib", "3",
                 "--min-swap-free-gib", "4"],
                ["conda", "run", "-n", "crystal_wsl", "python", str(converter), "convert-samples",
                 "--input", str(samples_dir / "prior_samples.csv"), "--output-dir", str(cifs_dir),
                 "--max-candidates", str(args.num_samples), "--require-formula-match"],
                ["conda", "run", "-n", "crystal_wsl", "python", str(evaluator),
                 "--target", str(target_dir / "target.csv"),
                 "--ground-truth", str(target_dir / "ground_truth.cif"),
                 "--candidates", str(cifs_dir), "--sampling-csv", str(samples_dir / "prior_samples.csv"),
                 "--formula", formula, "--output", str(run_output / "common_metrics.csv")],
            ]
            # Resume a partially completed evaluation without resampling.
            if (samples_dir / "prior_samples.csv").is_file():
                commands = commands[2:]
            elif params_checkpoint.is_file():
                commands = commands[1:]
            run_output.mkdir(parents=True, exist_ok=True)
            run_started = time.monotonic()
            status = "complete"
            return_code = 0
            for step, command in enumerate(commands, 1):
                return_code = _run(command, output_root / "logs" / f"{run_id}_step{step}.log", deadline - time.monotonic())
                if return_code != 0:
                    status = f"failed_step_{step}"
                    break
            record = {
                "run_id": run_id, "formula": formula, "training_method": method,
                "training_seed": training_seed, "test_seed": test_seed,
                "status": status, "return_code": return_code,
                "elapsed_seconds": time.monotonic() - run_started,
                "training_checkpoint": str(checkpoint), "output_dir": str(run_output),
            }
            manifest["runs"].append(record)
            _write_manifest(manifest_path, manifest)
            print(f"{run_id}: {status} ({record['elapsed_seconds']:.1f}s)", flush=True)
            if status != "complete":
                manifest.update({"status": "failed", "stop_reason": run_id})
                _write_manifest(manifest_path, manifest)
                return

    rows = []
    for record in manifest["runs"]:
        summary = json.loads((Path(record["output_dir"]) / "common_metrics.json").read_text())
        rows.append({**{key: record[key] for key in ("run_id", "formula", "training_method", "training_seed", "test_seed")}, **summary})
    with (output_root / "INDEPENDENT_EVALUATION_SUMMARY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest.update({"status": "complete", "elapsed_seconds": time.monotonic() - started})
    _write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
