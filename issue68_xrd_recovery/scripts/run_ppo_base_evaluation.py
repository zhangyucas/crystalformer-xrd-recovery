#!/usr/bin/env python
"""Evaluate the common pre-PPO checkpoint on the independent test seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_ppo_independent_evaluation import CASES, SEED_OFFSETS, _run


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=4.0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bridge = repo / "issue68_xrd_recovery/scripts/run_gpu_host.sh"
    sampler = repo / "issue68_xrd_recovery/scripts/xrd_prior_sample.py"
    converter = repo / "issue68_xrd_recovery/scripts/xrd_recovery.py"
    evaluator = repo / "issue68_xrd_recovery/scripts/evaluate_ppo_sample_pool.py"
    benchmark = repo / "issue68_xrd_recovery/experiments/benchmark_20260731/opxrd_subset"
    params_checkpoint = Path(args.checkpoint).resolve()
    deadline = time.monotonic() + args.max_hours * 3600
    manifest_path = root / "base_evaluation_manifest.json"
    manifest = {"status": "running", "samples_per_run": args.num_samples, "runs": []}
    _write(manifest_path, manifest)

    for formula, case_dir, test_seed_base in CASES:
        target_dir = benchmark / case_dir
        for seed_offset in SEED_OFFSETS:
            test_seed = test_seed_base + seed_offset
            run_id = f"{case_dir}_base_test{test_seed}"
            output = root / "runs" / run_id
            summary_path = output / "common_metrics.json"
            if summary_path.is_file():
                manifest["runs"].append({
                    "run_id": run_id, "formula": formula, "test_seed": test_seed,
                    "status": "skipped_complete", "output_dir": str(output),
                })
                _write(manifest_path, manifest)
                continue
            if deadline - time.monotonic() < 600:
                manifest["status"] = "deadline_stop"
                _write(manifest_path, manifest)
                return
            samples = output / "samples"
            candidates = output / "candidates"
            commands = [
                [str(bridge), "conda", "run", "-n", "crystal_wsl", "env",
                 "XLA_PYTHON_CLIENT_PREALLOCATE=false", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.40",
                 "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                 "python", str(sampler), "sample", "--checkpoint", str(params_checkpoint),
                 "--output-dir", str(samples), "--formula", formula, "--num-samples", str(args.num_samples),
                 "--batch-size", "2", "--seed", str(test_seed), "--platform", "gpu",
                 "--composition-max-atoms", "40", "--composition-size-bias", "0.5",
                 "--min-available-gib", "3", "--min-swap-free-gib", "4"],
                ["conda", "run", "-n", "crystal_wsl", "python", str(converter), "convert-samples",
                 "--input", str(samples / "prior_samples.csv"), "--output-dir", str(candidates),
                 "--max-candidates", str(args.num_samples), "--require-formula-match"],
                ["conda", "run", "-n", "crystal_wsl", "python", str(evaluator),
                 "--target", str(target_dir / "target.csv"), "--ground-truth", str(target_dir / "ground_truth.cif"),
                 "--candidates", str(candidates), "--sampling-csv", str(samples / "prior_samples.csv"),
                 "--formula", formula, "--output", str(output / "common_metrics.csv")],
            ]
            output.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            status = "complete"
            return_code = 0
            for step, command in enumerate(commands, 1):
                return_code = _run(command, root / "logs" / f"{run_id}_step{step}.log", deadline - time.monotonic())
                if return_code:
                    status = f"failed_step_{step}"
                    break
            record = {
                "run_id": run_id, "formula": formula, "test_seed": test_seed,
                "status": status, "return_code": return_code,
                "elapsed_seconds": time.monotonic() - started, "output_dir": str(output),
            }
            manifest["runs"].append(record)
            _write(manifest_path, manifest)
            print(f"{run_id}: {status} ({record['elapsed_seconds']:.1f}s)", flush=True)
            if status != "complete":
                manifest.update({"status": "failed", "stop_reason": run_id})
                _write(manifest_path, manifest)
                return

    rows = []
    for record in manifest["runs"]:
        summary = json.loads((Path(record["output_dir"]) / "common_metrics.json").read_text())
        rows.append({**{key: record[key] for key in ("run_id", "formula", "test_seed")}, **summary})
    with (root / "BASE_EVALUATION_SUMMARY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["status"] = "complete"
    _write(manifest_path, manifest)


if __name__ == "__main__":
    main()
