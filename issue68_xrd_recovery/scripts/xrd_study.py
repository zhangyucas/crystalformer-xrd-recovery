#!/usr/bin/env python
"""Create compact recovery and prior-strength (tau/beta) study reports.

The command only reads small JSON summaries and CSV metadata.  It never loads
CrystalFormer checkpoints, so it is safe to run after a sweep on a constrained
machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
from statistics import mean, pstdev


def _as_bool(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _run_config(summary_path: Path) -> dict:
    config_path = summary_path.parent / "run_config.json"
    candidates = [config_path]
    if not config_path.exists():
        nested = sorted(summary_path.parent.rglob("run_config.json"))
        if len(nested) == 1:
            candidates = nested
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _summary_rows(root: Path, threshold: float) -> list[dict]:
    rows = []
    summary_dirs = set()
    for summary_path in sorted(root.rglob("*summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = summary.get("run_config") if isinstance(summary.get("run_config"), dict) else _run_config(summary_path)
        nested_config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
        score = summary.get("best_score", summary.get("xrd_similarity"))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        tau = summary.get("tau", summary.get("beta", config.get("beta", config.get("tau"))))
        try:
            tau = float(tau) if tau is not None else None
        except (TypeError, ValueError):
            tau = None
        match = _as_bool(summary.get("structure_match"))
        rows.append({
            "path": str(summary_path),
            "method": summary.get("method", "unknown"),
            "formula": summary.get("formula", config.get("formula", "")),
            "seed": summary.get("seed", nested_config.get("seed", config.get("seed", ""))),
            "tau": tau,
            "best_similarity": score,
            "structure_match": "" if match is None else int(match),
            "similarity_recovered": "" if score is None else int(score >= threshold),
            "evaluations": summary.get("evaluations", ""),
        })
        summary_dirs.add(summary_path.parent.resolve())
        if summary.get("run_dir"):
            summary_dirs.add(Path(summary["run_dir"]).resolve())

    # PPO writes a compact whitespace log rather than a summary JSON.  Read
    # only its last/best scalar columns; checkpoints and candidate CIFs are
    # deliberately never loaded here.
    for config_path in sorted(root.rglob("run_config.json")):
        run_dir = config_path.parent.resolve()
        if run_dir in summary_dirs:
            continue
        data_path = run_dir / "data.txt"
        if not data_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            lines = [line.split() for line in data_path.read_text().splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("reward") != "xrd":
            continue
        if len(lines) < 2:
            continue
        header = lines[0]
        metric_name = "xrd"
        max_index = next((i for i, name in enumerate(header) if name == f"{metric_name}_max"), None)
        if max_index is None:
            continue
        values = []
        for values_line in lines[1:]:
            if len(values_line) <= max_index:
                continue
            try:
                values.append(float(values_line[max_index]))
            except ValueError:
                continue
        if not values:
            continue
        tau = config.get("beta", config.get("tau"))
        try:
            tau = float(tau) if tau is not None else None
        except (TypeError, ValueError):
            tau = None
        rows.append({
            "path": str(data_path),
            "method": "crystalformer_ppo",
            "formula": config.get("formula", ""),
            "seed": config.get("seed", ""),
            "tau": tau,
            "best_similarity": max(values),
            "structure_match": "",
            "similarity_recovered": int(max(values) >= threshold),
            "evaluations": len(values),
        })
    return rows


def _group_rows(rows: list[dict], threshold: float) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["method"], row["tau"])
        groups.setdefault(key, []).append(row)
    grouped = []
    for (method, tau), members in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or 0)):
        scores = [row["best_similarity"] for row in members if row["best_similarity"] is not None]
        matches = [row["structure_match"] for row in members if row["structure_match"] != ""]
        similarity_hits = [row for row in members if row["similarity_recovered"] == 1]
        grouped.append({
            "method": method,
            "tau": "" if tau is None else tau,
            "runs": len(members),
            "mean_similarity": mean(scores) if scores else "",
            "std_similarity": pstdev(scores) if len(scores) > 1 else 0.0 if scores else "",
            "structure_match_rate": (sum(matches) / len(matches)) if matches else "",
            "structure_match_n": len(matches),
            "similarity_recovery_rate": len(similarity_hits) / len(members) if members else "",
            "threshold": threshold,
        })
    return grouped


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.runs)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    rows = _summary_rows(root, args.threshold)
    if not rows:
        raise FileNotFoundError(f"no summary.json files found under {root}")
    grouped = _group_rows(rows, args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = output.with_name(output.stem + "_runs.csv")
    groups_path = output.with_suffix(".csv")
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with groups_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0].keys()))
        writer.writeheader()
        writer.writerows(grouped)

    report_path = output.with_suffix(".md")
    lines = [
        "# Powder XRD Recovery Study",
        "",
        f"Similarity recovery threshold: `{args.threshold:g}`.",
        "StructureMatcher recovery is reported separately and is only computed when a ground truth was supplied.",
        "",
        "| method | tau | runs | mean similarity | std similarity | StructureMatcher rate | similarity rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped:
        lines.append(
            "| {method} | {tau} | {runs} | {mean_similarity} | {std_similarity} | {structure_match_rate} ({structure_match_n}) | {similarity_recovery_rate} |".format(**row)
        )
    lines.extend([
        "",
        "The random-move baseline has no CrystalFormer prior. A tau sweep changes the PPO KL/prior term (`beta` is retained as the CLI spelling).",
        "Peak similarity is a fit metric; it is not by itself evidence that the recovered structure is the ground truth.",
    ])
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote run-level table: {rows_path}")
    print(f"Wrote grouped table: {groups_path}")
    print(f"Wrote report: {report_path}")


def make_commands(args: argparse.Namespace) -> None:
    taus = [float(value.strip()) for value in args.tau.split(",") if value.strip()]
    if not taus:
        raise ValueError("--tau must contain at least one value")
    if any(tau < 0 for tau in taus):
        raise ValueError("--tau values must be non-negative")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if args.evaluation_max_candidates <= 0:
        raise ValueError("--evaluation-max-candidates must be positive")
    if args.sg_temperature is not None and args.sg_temperature <= 0:
        raise ValueError("--sg-temperature must be positive")
    if not 0.0 <= args.sg_epsilon <= 1.0:
        raise ValueError("--sg-epsilon must be in [0, 1]")
    if args.diversity_weight < 0:
        raise ValueError("--diversity-weight cannot be negative")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    command_path = root / "run_tau_sweep.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "# Run one value at a time on a small WSL host."]
    for tau in taus:
        for seed in seeds:
            output = root / f"tau_{tau:g}" / f"seed_{seed}"
            command = [
                "env",
                "JAX_PLATFORMS=cpu",
                "JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
                "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "MPLCONFIGDIR=/tmp/crystalformer-mpl",
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "python",
                "-m",
                "crystalformer.cli.train_ppo",
                "--reward",
                "xrd",
                "--formula",
                args.formula,
                "--xrd_target",
                args.target,
                "--restore_path",
                args.restore_path,
                "--folder",
                str(output),
                "--tau",
                str(tau),
                "--safe_cpu",
                "--epochs",
                str(args.epochs),
                "--ppo_epochs",
                str(args.ppo_epochs),
                "--batchsize",
                str(args.batchsize),
                "--sample_multiplier",
                str(args.sample_multiplier),
                "--max_sampling_attempts",
                str(args.max_sampling_attempts),
                "--seed",
                str(seed),
                "--h0_size",
                str(args.h0_size),
                "--transformer_layers",
                str(args.transformer_layers),
                "--num_heads",
                str(args.num_heads),
                "--key_size",
                str(args.key_size),
                "--model_size",
                str(args.model_size),
                "--embed_size",
                str(args.embed_size),
            ]
            if args.sg_temperature is not None:
                command.extend(["--sg_temperature", str(args.sg_temperature)])
            if args.sg_epsilon > 0:
                command.extend(["--sg_epsilon", str(args.sg_epsilon)])
            if args.diversity_weight > 0:
                command.extend(["--diversity_weight", str(args.diversity_weight)])
            lines.append(" \\\n  ".join(shlex.quote(item) for item in command))
            if args.ground_truth:
                evaluation_command = [
                    "python",
                    "issue68_xrd_recovery/scripts/xrd_recovery.py",
                    "evaluate",
                    "--target",
                    args.target,
                    "--candidates",
                    str(output),
                    "--ground-truth",
                    args.ground_truth,
                    "--output",
                    str(output / "evaluation.csv"),
                    "--max-candidates",
                    str(args.evaluation_max_candidates),
                ]
                lines.append(" \\\n  ".join(shlex.quote(item) for item in evaluation_command))
            lines.append("")
    command_path.write_text("\n".join(lines))
    command_path.chmod(0o755)
    print(f"Wrote tau sweep commands: {command_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--runs", required=True)
    summary.add_argument("--output", required=True, help="output stem, e.g. experiments/xrd/study")
    summary.add_argument("--threshold", type=float, default=0.9)
    summary.set_defaults(func=summarize)

    commands = subparsers.add_parser("make-commands")
    commands.add_argument("--target", required=True)
    commands.add_argument("--formula", required=True)
    commands.add_argument("--restore-path", required=True)
    commands.add_argument("--output-root", required=True)
    commands.add_argument("--tau", default="0,0.01,0.1,0.5")
    commands.add_argument("--seeds", default="42",
                          help="comma-separated seeds; commands remain serial")
    commands.add_argument("--ground-truth", default=None,
                          help="optional CIF; append StructureMatcher evaluation after each PPO run")
    commands.add_argument("--evaluation-max-candidates", type=int, default=100)
    commands.add_argument("--epochs", type=int, default=1)
    commands.add_argument("--ppo-epochs", type=int, default=1)
    commands.add_argument("--batchsize", type=int, default=4)
    commands.add_argument("--sample-multiplier", type=float, default=2)
    commands.add_argument("--max-sampling-attempts", type=int, default=20)
    commands.add_argument("--h0-size", type=int, default=64)
    commands.add_argument("--transformer-layers", type=int, default=2)
    commands.add_argument("--num-heads", type=int, default=4)
    commands.add_argument("--key-size", type=int, default=16)
    commands.add_argument("--model-size", type=int, default=64)
    commands.add_argument("--embed-size", type=int, default=64)
    commands.add_argument("--sg-temperature", type=float, default=None)
    commands.add_argument("--sg-epsilon", type=float, default=0.0)
    commands.add_argument("--diversity-weight", type=float, default=0.0)
    commands.set_defaults(func=make_commands)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
