#!/usr/bin/env python
"""Sample a pretrained CrystalFormer prior with a bounded host-memory peak.

The GPU path intentionally builds only the Transformer apply function and the
sampler.  It does not construct a loss, log-probability function, optimizer, or
gradient state.  Use ``extract-params`` once to strip the unused optimizer state
from a training checkpoint before starting the GPU process.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import pickle
import signal
import threading
import time
from typing import Any, Sequence

# Set thread bounds before NumPy or JAX can initialize a BLAS runtime.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np


GIB = 1024 ** 3


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    return values


def _process_rss_bytes() -> int:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class MemoryMonitor:
    """Persist telemetry and terminate before WSL memory is exhausted."""

    def __init__(
        self,
        output_dir: Path,
        min_available_gib: float,
        min_swap_free_gib: float,
        interval: float,
    ) -> None:
        self.output_dir = output_dir
        self.min_available_bytes = int(min_available_gib * GIB)
        self.min_swap_free_bytes = int(min_swap_free_gib * GIB)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.min_available_bytes_seen: int | None = None
        self.min_swap_free_bytes_seen: int | None = None
        self.max_rss_bytes_seen = 0

    def start(self) -> None:
        telemetry_path = self.output_dir / "memory_telemetry.csv"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "unix_time",
                "mem_available_mib",
                "swap_free_mib",
                "process_rss_mib",
            ])
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval * 2.0))

    def _run(self) -> None:
        telemetry_path = self.output_dir / "memory_telemetry.csv"
        while not self.stop_event.is_set():
            memory = _read_meminfo()
            available = memory.get("MemAvailable", 0)
            swap_free = memory.get("SwapFree", 0)
            rss = _process_rss_bytes()
            self.min_available_bytes_seen = (
                available
                if self.min_available_bytes_seen is None
                else min(self.min_available_bytes_seen, available)
            )
            self.min_swap_free_bytes_seen = (
                swap_free
                if self.min_swap_free_bytes_seen is None
                else min(self.min_swap_free_bytes_seen, swap_free)
            )
            self.max_rss_bytes_seen = max(self.max_rss_bytes_seen, rss)
            with telemetry_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([
                    f"{time.time():.3f}",
                    f"{available / (1024 ** 2):.1f}",
                    f"{swap_free / (1024 ** 2):.1f}",
                    f"{rss / (1024 ** 2):.1f}",
                ])
                handle.flush()

            below_memory = available < self.min_available_bytes
            below_swap = swap_free < self.min_swap_free_bytes
            if below_memory or below_swap:
                reason = {
                    "reason": "memory_stop",
                    "mem_available_gib": available / GIB,
                    "swap_free_gib": swap_free / GIB,
                    "min_available_gib": self.min_available_bytes / GIB,
                    "min_swap_free_gib": self.min_swap_free_bytes / GIB,
                    "unix_time": time.time(),
                }
                (self.output_dir / "stop_reason.json").write_text(
                    json.dumps(reason, indent=2, sort_keys=True) + "\n"
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self.stop_event.wait(self.interval)

    def summary(self) -> dict[str, float | None]:
        return {
            "min_available_gib_seen": (
                None
                if self.min_available_bytes_seen is None
                else self.min_available_bytes_seen / GIB
            ),
            "min_swap_free_gib_seen": (
                None
                if self.min_swap_free_bytes_seen is None
                else self.min_swap_free_bytes_seen / GIB
            ),
            "max_process_rss_gib_seen": self.max_rss_bytes_seen / GIB,
        }


def _json_array(value: Any) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def _actual_composition(
    atoms: Sequence[int], multiplicities: Sequence[int], atom_types: int
) -> np.ndarray:
    composition = np.zeros(atom_types, dtype=np.int64)
    for atom, multiplicity in zip(atoms, multiplicities):
        atom = int(atom)
        if 0 < atom < atom_types:
            composition[atom] += int(multiplicity)
    nonzero = composition[composition > 0]
    if nonzero.size:
        divisor = int(np.gcd.reduce(nonzero))
        if divisor > 1:
            composition //= divisor
    return composition


def _sample_row(
    sample_index: int,
    batch_index: int,
    target_composition: Sequence[int],
    sample: Sequence[Any],
) -> dict[str, Any]:
    g, xyz, atoms, wyckoff, multiplicities, lattice = sample
    lattice = np.asarray(lattice, dtype=np.float64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64)
    atoms = np.asarray(atoms, dtype=np.int64).reshape(-1)
    wyckoff = np.asarray(wyckoff, dtype=np.int64).reshape(-1)
    multiplicities = np.asarray(multiplicities, dtype=np.int64).reshape(-1)
    target = np.asarray(target_composition, dtype=np.int64).reshape(-1)
    actual = _actual_composition(atoms, multiplicities, target.size)
    finite = bool(np.isfinite(lattice).all() and np.isfinite(xyz).all())
    lattice_valid = bool(
        finite
        and lattice.size == 6
        and np.all((lattice[:3] >= 1e-4) & (lattice[:3] <= 1e4))
        and np.all((lattice[3:] > 0.0) & (lattice[3:] < 180.0))
    )
    nonempty = bool(np.sum(multiplicities) > 0 and np.any(atoms > 0))
    return {
        "sample_index": sample_index,
        "batch_index": batch_index,
        "G": int(np.asarray(g)),
        "L": _json_array(lattice),
        "XYZ": _json_array(xyz),
        "A": _json_array(atoms),
        "W": _json_array(wyckoff),
        "M": _json_array(multiplicities),
        "num_atoms": int(np.sum(multiplicities)),
        "formula_match": bool(np.array_equal(actual, target)),
        "finite": finite,
        "lattice_valid": lattice_valid,
        "geometry_precheck": bool(lattice_valid and nonempty),
    }


def _load_params(path: Path) -> Any:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"checkpoint does not contain a params tree: {path}")
    return payload["params"]


def extract_params(args: argparse.Namespace) -> None:
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    if source == output:
        raise ValueError("--output must differ from --checkpoint")
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    params = _load_params(source)
    payload = {
        "params": params,
        "source_checkpoint": str(source),
        "params_only": True,
    }
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    print(f"Wrote params-only checkpoint: {output}")
    print(f"Source size: {source.stat().st_size / (1024 ** 2):.1f} MiB")
    print(f"Params-only size: {output.stat().st_size / (1024 ** 2):.1f} MiB")


def sample_prior(args: argparse.Namespace) -> None:
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive")
    if args.min_available_gib <= 0 or args.min_swap_free_gib < 0:
        raise ValueError("memory stop thresholds must be non-negative")
    if args.monitor_interval <= 0:
        raise ValueError("--monitor-interval must be positive")
    if not 0.0 < args.top_p <= 1.0 or args.temperature <= 0:
        raise ValueError("--top-p must be in (0, 1] and --temperature must be positive")
    if args.sg_temperature is not None and args.sg_temperature <= 0:
        raise ValueError("--sg-temperature must be positive")
    if not 0.0 <= args.sg_epsilon <= 1.0:
        raise ValueError("--sg-epsilon must be in [0, 1]")
    if args.K < 0 or args.K > 230:
        raise ValueError("--K must be in [0, 230]")
    if args.spacegroup is not None and not 1 <= args.spacegroup <= 230:
        raise ValueError("--spacegroup must be in [1, 230]")

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "prior_samples.csv"
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"sample output already exists: {output_csv}; pass --overwrite for a new run"
        )

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
    if args.compilation_cache:
        cache_path = Path(args.compilation_cache).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_path))

    run_config = vars(args).copy()
    run_config.pop("func", None)
    run_config.update({
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "checkpoint_size_mib": checkpoint_path.stat().st_size / (1024 ** 2),
        "xla_python_client_preallocate": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "xla_python_client_mem_fraction": os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
    })
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    monitor = MemoryMonitor(
        output_dir,
        args.min_available_gib,
        args.min_swap_free_gib,
        args.monitor_interval,
    )
    monitor.start()
    started = time.monotonic()
    try:
        import jax

        jax.config.update("jax_enable_x64", bool(args.enable_x64))
        if args.platform != "auto":
            jax.config.update("jax_platform_name", args.platform)
        import jax.numpy as jnp

        from crystalformer.src.formula import formula_string_to_composition_vector
        from crystalformer.src.sample import make_sample_crystal
        from crystalformer.src.transformer import make_transformer

        devices = jax.devices()
        if not devices:
            raise RuntimeError("JAX did not find a compute device")
        print(f"JAX backend: {jax.default_backend()}", flush=True)
        print(f"JAX devices: {devices}", flush=True)

        key = jax.random.PRNGKey(args.seed)
        _, transformer = make_transformer(
            key,
            args.Nf,
            args.Kx,
            args.Kl,
            args.n_max,
            args.h0_size,
            args.transformer_layers,
            args.num_heads,
            args.key_size,
            args.model_size,
            args.embed_size,
            args.atom_types,
            args.wyck_types,
            args.dropout_rate,
            args.attn_dropout,
            initialize=False,
        )
        params = _load_params(checkpoint_path)
        params = jax.device_put(params, devices[0])
        gc.collect()
        composition = formula_string_to_composition_vector(args.formula)
        target_composition = np.asarray(jax.device_get(composition), dtype=np.int64)
        composition = jnp.asarray(composition)
        sampler = make_sample_crystal(
            transformer,
            args.n_max,
            args.atom_types,
            args.wyck_types,
            args.Kx,
            args.Kl,
            None,
            args.top_p,
            args.temperature,
            K=args.K,
            g=args.spacegroup,
            sg_temperature=args.sg_temperature,
            sg_epsilon=args.sg_epsilon,
        )

        fieldnames = [
            "sample_index",
            "batch_index",
            "G",
            "L",
            "XYZ",
            "A",
            "W",
            "M",
            "num_atoms",
            "formula_match",
            "finite",
            "lattice_valid",
            "geometry_precheck",
        ]
        formula_matches = 0
        geometry_valid = 0
        samples_written = 0
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            handle.flush()
            for batch_index in range(num_batches):
                key, sample_key = jax.random.split(key)
                batch_started = time.monotonic()
                # Keep a fixed static shape for every call.  A smaller final
                # batch would force a second high-memory XLA compilation.
                arrays = sampler(sample_key, params, args.batch_size, composition)
                g, xyz, atoms, wyckoff, multiplicities, lattice = jax.device_get(arrays)
                remaining = args.num_samples - samples_written
                keep = min(args.batch_size, remaining)
                for offset in range(keep):
                    row = _sample_row(
                        samples_written,
                        batch_index,
                        target_composition,
                        (
                            g[offset],
                            xyz[offset],
                            atoms[offset],
                            wyckoff[offset],
                            multiplicities[offset],
                            lattice[offset],
                        ),
                    )
                    writer.writerow(row)
                    formula_matches += int(row["formula_match"])
                    geometry_valid += int(row["geometry_precheck"])
                    samples_written += 1
                handle.flush()
                print(
                    f"batch {batch_index + 1}/{num_batches}: wrote {samples_written}/{args.num_samples} "
                    f"samples in {time.monotonic() - batch_started:.1f}s",
                    flush=True,
                )

        jax.block_until_ready(arrays)
        summary = {
            "method": "crystalformer_prior_sampling",
            "formula": args.formula,
            "seed": args.seed,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "requested_samples": args.num_samples,
            "samples_written": samples_written,
            "batch_size": args.batch_size,
            "compiled_batch_shape": args.batch_size,
            "formula_matches": formula_matches,
            "geometry_precheck_pass": geometry_valid,
            "elapsed_seconds": time.monotonic() - started,
            **monitor.summary(),
        }
        (output_dir / "sampling_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"Wrote samples: {output_csv}", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
            **monitor.summary(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise
    finally:
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract-params", help="write a params-only checkpoint in an isolated process"
    )
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--force", action="store_true")
    extract.set_defaults(func=extract_params)

    sample = subparsers.add_parser("sample", help="sample the pretrained prior")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--formula", required=True)
    sample.add_argument("--num-samples", type=int, default=20)
    sample.add_argument("--batch-size", type=int, default=1)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--platform", choices=["auto", "cpu", "gpu"], default="auto")
    sample.add_argument("--top-p", type=float, default=1.0)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--sg-temperature", type=float, default=None)
    sample.add_argument("--sg-epsilon", type=float, default=0.0)
    sample.add_argument("--K", type=int, default=0)
    sample.add_argument("--spacegroup", type=int, default=None)
    sample.add_argument("--compilation-cache", default=None)
    sample.add_argument("--min-available-gib", type=float, default=1.5)
    sample.add_argument("--min-swap-free-gib", type=float, default=0.5)
    sample.add_argument("--monitor-interval", type=float, default=0.5)
    sample.add_argument("--enable-x64", action="store_true")
    sample.add_argument("--overwrite", action="store_true")

    sample.add_argument("--Nf", type=int, default=5)
    sample.add_argument("--Kx", type=int, default=16)
    sample.add_argument("--Kl", type=int, default=4)
    sample.add_argument("--n-max", type=int, default=21)
    sample.add_argument("--atom-types", type=int, default=119)
    sample.add_argument("--wyck-types", type=int, default=28)
    sample.add_argument("--h0-size", type=int, default=256)
    sample.add_argument("--transformer-layers", type=int, default=16)
    sample.add_argument("--num-heads", type=int, default=8)
    sample.add_argument("--key-size", type=int, default=32)
    sample.add_argument("--model-size", type=int, default=256)
    sample.add_argument("--embed-size", type=int, default=256)
    sample.add_argument("--dropout-rate", type=float, default=0.1)
    sample.add_argument("--attn-dropout", type=float, default=0.1)
    sample.set_defaults(func=sample_prior)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
