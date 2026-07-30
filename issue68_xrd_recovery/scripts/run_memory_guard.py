#!/usr/bin/env python
"""Run one command with WSL memory telemetry and hard stop thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GIB = 1024 ** 3


def _meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    return values


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("a command is required after --")
    if args.min_available_gib <= 0 or args.min_swap_free_gib < 0:
        raise ValueError("memory thresholds must be non-negative")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / "guard_memory_telemetry.csv"
    summary_path = output_dir / "guard_summary.json"
    stop_path = output_dir / "guard_stop_reason.json"
    started = time.monotonic()
    minimum_available = None
    minimum_swap = None
    maximum_rss = 0
    stopped = False

    with telemetry_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "unix_time",
            "mem_available_mib",
            "swap_free_mib",
            "child_rss_mib",
        ])
        handle.flush()
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None:
            memory = _meminfo()
            available = memory.get("MemAvailable", 0)
            swap_free = memory.get("SwapFree", 0)
            rss = _rss_bytes(process.pid)
            minimum_available = available if minimum_available is None else min(minimum_available, available)
            minimum_swap = swap_free if minimum_swap is None else min(minimum_swap, swap_free)
            maximum_rss = max(maximum_rss, rss)
            writer.writerow([
                f"{time.time():.3f}",
                f"{available / (1024 ** 2):.1f}",
                f"{swap_free / (1024 ** 2):.1f}",
                f"{rss / (1024 ** 2):.1f}",
            ])
            handle.flush()
            if (
                available < args.min_available_gib * GIB
                or swap_free < args.min_swap_free_gib * GIB
            ):
                stopped = True
                reason = {
                    "reason": "memory_stop",
                    "command": command,
                    "mem_available_gib": available / GIB,
                    "swap_free_gib": swap_free / GIB,
                    "min_available_gib": args.min_available_gib,
                    "min_swap_free_gib": args.min_swap_free_gib,
                }
                stop_path.write_text(json.dumps(reason, indent=2, sort_keys=True) + "\n")
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(args.interval)

    return_code = process.returncode
    summary = {
        "command": command,
        "return_code": return_code,
        "memory_stopped": stopped,
        "elapsed_seconds": time.monotonic() - started,
        "min_available_gib_seen": None if minimum_available is None else minimum_available / GIB,
        "min_swap_free_gib_seen": None if minimum_swap is None else minimum_swap / GIB,
        "max_child_rss_gib_seen": maximum_rss / GIB,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 75 if stopped else int(return_code or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--min-swap-free-gib", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
