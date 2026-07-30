#!/usr/bin/env python
"""Generate reproducible XRD targets and evaluate candidate CIFs.

This script intentionally evaluates candidates one at a time.  It is useful
for CPU-only smoke tests and for post-processing CrystalFormer/PPO outputs
without loading a large candidate database into memory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

from crystalformer.reinforce.xrd import (
    XRDConfig,
    config_dict,
    cosine_similarity,
    make_xrd_reward_fn,
    make_two_theta_grid,
    save_pattern,
    simulate_structure_pattern,
    structure_from_GLXYZAW,
)
from crystalformer.reinforce.xrd_baseline import RandomMoveConfig, run_random_move, structure_matches


def _wavelength(value: str) -> str | float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _bool_or_none(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _add_xrd_options(
    parser: argparse.ArgumentParser, *, metadata_defaults: bool = False
) -> None:
    default = None if metadata_defaults else argparse.SUPPRESS
    parser.add_argument("--wavelength", default=default)
    parser.add_argument("--two-theta-min", type=float, default=default)
    parser.add_argument("--two-theta-max", type=float, default=default)
    parser.add_argument("--grid-step", type=float, default=default)
    parser.add_argument("--profile", choices=["gaussian", "pseudo-voigt"], default=default)
    parser.add_argument("--fwhm", type=float, default=default)
    parser.add_argument("--eta", type=float, default=default)
    parser.add_argument("--target-is-peaks", action="store_true", default=None if metadata_defaults else False)


def _config(args: argparse.Namespace, metadata: dict | None = None) -> XRDConfig:
    metadata = metadata or {}
    def choose(name: str, fallback):
        value = getattr(args, name, None)
        return metadata.get(name, fallback) if value is None else value

    wavelength = choose("wavelength", "CuKa")
    two_theta_min = choose("two_theta_min", 5.0)
    two_theta_max = choose("two_theta_max", 90.0)
    grid_step = choose("grid_step", 0.05)
    profile = choose("profile", "gaussian")
    fwhm = choose("fwhm", 0.10)
    eta = choose("eta", 0.5)
    target_is_peaks = choose("target_is_peaks", metadata.get("target_is_peaks", False))
    return XRDConfig(
        wavelength=_wavelength(wavelength),
        two_theta_min=float(two_theta_min),
        two_theta_max=float(two_theta_max),
        grid_step=float(grid_step),
        profile=profile,
        fwhm=float(fwhm),
        eta=float(eta),
        target_is_peaks=bool(_bool_or_none(target_is_peaks)),
    )


def _metadata_for(path: Path) -> dict:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read target metadata {metadata_path}: {exc}") from exc


def _run_config_for(candidates: str | Path) -> tuple[dict, Path | None]:
    """Find a nearby PPO run description without loading model artifacts."""

    root = Path(candidates)
    probes = []
    if root.is_dir():
        probes.append(root)
    else:
        probes.append(root.parent)
    probes.extend(root.parents)
    seen = set()
    for directory in probes:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        config_path = directory / "run_config.json"
        if config_path.is_file():
            try:
                return json.loads(config_path.read_text()), directory
            except (OSError, json.JSONDecodeError):
                return {}, None

    # A generated tau command keeps one run under each seed directory.  When
    # the evaluator is pointed at that directory, accept its unique nested
    # config; refuse to guess if several runs are mixed together.
    if root.is_dir():
        nested = sorted(root.rglob("run_config.json"))
        if len(nested) == 1:
            try:
                return json.loads(nested[0].read_text()), nested[0].parent.resolve()
            except (OSError, json.JSONDecodeError):
                pass
    return {}, None


def generate_target(args: argparse.Namespace) -> None:
    if args.noise_std < 0:
        raise ValueError("noise standard deviation cannot be negative")
    config = _config(args)
    structure = Structure.from_file(args.cif)
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    grid = make_two_theta_grid(config.two_theta_range, config.grid_step)
    _, clean_curve = simulate_structure_pattern(
        structure,
        XRDCalculator(wavelength=config.wavelength),
        config,
        grid,
    )
    scale = float(clean_curve.max())
    if scale <= 0:
        raise ValueError("target structure has no positive XRD intensity")
    curve = clean_curve / scale
    rng = np.random.default_rng(args.seed)
    if args.noise_std > 0:
        curve = np.maximum(curve + rng.normal(0.0, args.noise_std, size=curve.shape), 0.0)

    output = Path(args.output)
    metadata = config_dict(config)
    metadata.update({
        "source_cif": str(Path(args.cif).resolve()),
        "noise_std": args.noise_std,
        "seed": args.seed,
        "normalized_peak_height": scale,
    })
    save_pattern(output, grid, curve, metadata=metadata)
    print(f"Wrote target pattern: {output}")
    print(f"Grid points: {len(grid)}, clean peak scale: {scale:.6g}")


def _candidate_paths(root: str, max_candidates: int) -> list[Path]:
    def is_structure_file(item: Path) -> bool:
        if not item.is_file() or item.suffix.lower() not in {".cif", ".json"}:
            return False
        # Pattern sidecars, PPO run metadata, and evaluation summaries are
        # JSON but are not crystal structures.  Keeping this filter here
        # prevents a recursive study evaluation from scoring its own metadata.
        name = item.name.lower()
        return not (
            name == "run_config.json"
            or name.endswith(".csv.json")
            or name.endswith("_summary.json")
            or name.endswith("summary.json")
        )

    path = Path(root)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            item for item in path.rglob("*")
            if is_structure_file(item)
        )
    else:
        candidates = sorted(
            Path(item) for item in glob.glob(root, recursive=True)
            if is_structure_file(Path(item))
        )
    candidates = [p for p in candidates if is_structure_file(p)]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def _serialized_array(value: str) -> np.ndarray:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return np.asarray(parsed)


def convert_samples(args: argparse.Namespace) -> None:
    """Convert a streamed prior-sampling CSV into individual candidate CIFs."""

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"sample CSV does not exist: {input_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    converted = 0
    with input_path.open(newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), 1):
            if args.max_candidates > 0 and converted >= args.max_candidates:
                break
            formula_match = _bool_or_none(row.get("formula_match"))
            if args.require_formula_match and formula_match is not True:
                continue
            result = {
                "source_row": source_row,
                "sample_index": row.get("sample_index", source_row - 1),
                "formula_match": formula_match,
                "cif": "",
                "formula": "",
                "num_sites": "",
                "error": "",
            }
            try:
                structure = structure_from_GLXYZAW(
                    int(row["G"]),
                    _serialized_array(row["L"]),
                    _serialized_array(row["XYZ"]),
                    _serialized_array(row["A"]),
                    _serialized_array(row["W"]),
                )
                sample_index = int(row.get("sample_index", source_row - 1))
                cif_path = output_dir / f"sample_{sample_index:04d}.cif"
                structure.to(filename=str(cif_path))
                result["cif"] = str(cif_path)
                result["formula"] = structure.composition.reduced_formula
                result["num_sites"] = len(structure)
                converted += 1
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(result)

    summary_path = output_dir / "conversion.csv"
    fieldnames = [
        "source_row",
        "sample_index",
        "formula_match",
        "cif",
        "formula",
        "num_sites",
        "error",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Converted {converted} candidates to {output_dir}")
    print(f"Wrote conversion summary: {summary_path}")


def evaluate_candidates(args: argparse.Namespace) -> None:
    target_path = Path(args.target)
    if not target_path.is_file():
        raise FileNotFoundError(f"target pattern/CIF does not exist: {target_path}")
    config = _config(args, _metadata_for(target_path))
    run_config, run_dir = _run_config_for(args.candidates)
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calculator = XRDCalculator(wavelength=config.wavelength)
    xrd_kwargs = {
        "wavelength": config.wavelength,
        "two_theta_range": config.two_theta_range,
        "grid_step": config.grid_step,
        "profile": config.profile,
        "fwhm": config.fwhm,
        "eta": config.eta,
        "target_is_peaks": config.target_is_peaks,
    }
    if target_path.suffix.lower() in {".cif", ".json"}:
        _, batch_reward = make_xrd_reward_fn(target_structure=target_path, **xrd_kwargs)
    else:
        _, batch_reward = make_xrd_reward_fn(target=target_path, **xrd_kwargs)
    grid = batch_reward.grid
    target_curve = batch_reward.target_curve
    if args.ground_truth and not Path(args.ground_truth).is_file():
        raise FileNotFoundError(f"ground-truth structure does not exist: {args.ground_truth}")
    ground_truth = Structure.from_file(args.ground_truth) if args.ground_truth else None
    matcher = None
    if ground_truth is not None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher()

    candidate_paths = _candidate_paths(args.candidates, args.max_candidates)
    if not candidate_paths:
        if not Path(args.candidates).exists() and not glob.glob(args.candidates, recursive=True):
            raise FileNotFoundError(f"no candidate path found: {args.candidates}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "path", "xrd_similarity", "structure_match", "formula", "num_sites", "error"])
        evaluation_summary = {
            "method": "crystalformer_ppo" if run_config.get("reward") == "xrd" else "candidate_evaluation",
            "target": str(target_path.resolve()),
            "evaluations": 0,
            "best_score": 0.0,
            "structure_match": None,
            "structure_match_count": 0 if matcher is not None else None,
            "structure_match_rate": 0.0 if matcher is not None else None,
            "first_structure_match_rank": None,
            "top_k_structure_match": None if matcher is None else False,
            "best_candidate": None,
        }
        if run_config:
            evaluation_summary["run_config"] = run_config
            evaluation_summary["formula"] = run_config.get("formula", "")
            evaluation_summary["seed"] = run_config.get("seed", "")
            evaluation_summary["tau"] = run_config.get("beta", run_config.get("tau", ""))
        if run_dir is not None:
            evaluation_summary["run_dir"] = str(run_dir)
        summary_path = output.with_name(output.stem + "_summary.json")
        summary_path.write_text(json.dumps(evaluation_summary, indent=2, sort_keys=True))
        print(f"No candidate CIFs found under {args.candidates}; wrote an empty evaluation")
        print(f"Wrote evaluation summary: {summary_path}")
        return

    rows = []
    for candidate_path in candidate_paths:
        row = {
            "path": str(candidate_path),
            "xrd_similarity": 0.0,
            "structure_match": "",
            "formula": "",
            "num_sites": "",
            "error": "",
        }
        try:
            structure = Structure.from_file(candidate_path)
            _, curve = simulate_structure_pattern(
                structure,
                calculator,
                config,
                grid,
            )
            row["xrd_similarity"] = cosine_similarity(curve, target_curve)
            row["formula"] = structure.composition.reduced_formula
            row["num_sites"] = len(structure)
            if matcher is not None:
                row["structure_match"] = bool(matcher.fit(structure, ground_truth))
        except Exception as exc:  # keep one malformed CIF from stopping a sweep
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    rows.sort(key=lambda row: float(row["xrd_similarity"]), reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        fieldnames = ["rank", *rows[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})

    top_row = rows[0]
    is_ppo_run = run_config.get("reward") == "xrd"
    matched_ranks = [
        rank
        for rank, row in enumerate(rows, 1)
        if _bool_or_none(row.get("structure_match")) is True
    ]
    evaluation_summary = {
        "method": "crystalformer_ppo" if is_ppo_run else "candidate_evaluation",
        "target": str(target_path.resolve()),
        "evaluations": len(rows),
        "best_score": float(top_row["xrd_similarity"]),
        "structure_match": (
            _bool_or_none(top_row.get("structure_match"))
            if matcher is not None else None
        ),
        "structure_match_count": len(matched_ranks) if matcher is not None else None,
        "structure_match_rate": (
            len(matched_ranks) / len(rows) if matcher is not None else None
        ),
        "first_structure_match_rank": (
            matched_ranks[0] if matched_ranks else None
        ),
        "top_k_structure_match": (
            bool(matched_ranks and matched_ranks[0] <= args.top_k)
            if matcher is not None else None
        ),
        "best_candidate": top_row["path"],
    }
    if run_config:
        evaluation_summary["run_config"] = run_config
        evaluation_summary["formula"] = run_config.get("formula", "")
        evaluation_summary["seed"] = run_config.get("seed", "")
        evaluation_summary["tau"] = run_config.get("beta", run_config.get("tau", ""))
    if run_dir is not None:
        evaluation_summary["run_dir"] = str(run_dir)
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(evaluation_summary, indent=2, sort_keys=True))

    print(f"Evaluated {len(rows)} candidates")
    print(f"Wrote results: {output}")
    print(f"Wrote evaluation summary: {summary_path}")
    for rank, row in enumerate(rows[: args.top_k], 1):
        print(f"{rank:3d} {float(row['xrd_similarity']):.6f} {row['path']}")

    if args.plot and rows:
        _plot_top_candidates(rows[: args.top_k], target_path, grid, target_curve, config, args.plot)


def _plot_top_candidates(rows, target_path, grid, target_curve, config, output_path):
    import matplotlib.pyplot as plt
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(grid, target_curve, color="black", linewidth=1.5, label="target")
    for row in rows:
        try:
            structure = Structure.from_file(row["path"])
            _, curve = simulate_structure_pattern(
                structure,
                XRDCalculator(wavelength=config.wavelength),
                config,
                grid,
            )
            axis.plot(grid, curve, linewidth=0.8, label=f"{row['xrd_similarity']:.3f}")
        except Exception:
            continue
    axis.set(xlabel="2theta (degree)", ylabel="intensity (a.u.)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Wrote plot: {output_path}")


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("baseline produced an empty trace")
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_baseline(args: argparse.Namespace) -> None:
    """Run the no-prior random-move reference search."""

    target_path = Path(args.target)
    if not target_path.is_file():
        raise FileNotFoundError(f"target pattern/CIF does not exist: {target_path}")
    if args.ground_truth and not Path(args.ground_truth).is_file():
        raise FileNotFoundError(f"ground-truth structure does not exist: {args.ground_truth}")
    metadata = _metadata_for(target_path)
    config = _config(args, metadata)
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calculator = XRDCalculator(wavelength=config.wavelength)
    xrd_kwargs = {
        "wavelength": config.wavelength,
        "two_theta_range": config.two_theta_range,
        "grid_step": config.grid_step,
        "profile": config.profile,
        "fwhm": config.fwhm,
        "eta": config.eta,
        "target_is_peaks": config.target_is_peaks,
    }
    if target_path.suffix.lower() in {".cif", ".json"}:
        _, batch_reward = make_xrd_reward_fn(target_structure=target_path, **xrd_kwargs)
    else:
        _, batch_reward = make_xrd_reward_fn(target=target_path, **xrd_kwargs)
    grid = batch_reward.grid
    target_curve = batch_reward.target_curve

    def score(structure: Structure) -> float:
        _, curve = simulate_structure_pattern(structure, calculator, config, grid)
        return cosine_similarity(curve, target_curve)

    search_config = RandomMoveConfig(
        evaluations=args.evaluations,
        restarts=args.restarts,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        coordinate_sigma=args.coordinate_sigma,
        lattice_sigma=args.lattice_sigma,
        min_distance=args.min_distance,
        volume_per_atom_min=args.volume_per_atom_min,
        volume_per_atom_max=args.volume_per_atom_max,
        length_min=args.length_min,
        length_max=args.length_max,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        seed=args.seed,
    )
    result = run_random_move(args.formula, score, search_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.cif"
    result["best_structure"].to(fmt="cif", filename=str(best_path))
    _write_trace(output_dir / "trace.csv", result["records"])

    match = None
    if args.ground_truth:
        reference = Structure.from_file(args.ground_truth)
        match = structure_matches(result["best_structure"], reference)
    summary = {
        "method": "random_move_monte_carlo",
        "formula": args.formula,
        "best_score": result["best_score"],
        "evaluations": result["evaluations"],
        "attempts": result["attempts"],
        "structure_match": match,
        "target": str(target_path.resolve()),
        "best_cif": str(best_path.resolve()),
        "config": result["config"],
        "xrd_config": config_dict(config),
        "restarts": result["restarts"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if args.plot:
        _plot_top_candidates(
            [{"path": str(best_path), "xrd_similarity": result["best_score"]}],
            target_path,
            grid,
            target_curve,
            config,
            args.plot,
        )
    print(f"Random-move evaluations: {result['evaluations']}")
    print(f"Best cosine similarity: {result['best_score']:.6f}")
    if match is not None:
        print(f"StructureMatcher match: {match}")
    print(f"Wrote baseline outputs: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser("target", help="simulate a CIF into a reproducible target pattern")
    target.add_argument("--cif", required=True)
    target.add_argument("--output", required=True)
    target.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise standard deviation after max-normalization")
    target.add_argument("--seed", type=int, default=0)
    _add_xrd_options(target)
    target.set_defaults(func=generate_target)

    evaluate = subparsers.add_parser("evaluate", help="score candidate CIFs against a target CSV pattern or target CIF")
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--candidates", required=True, help="CIF file, directory, or glob")
    evaluate.add_argument("--ground-truth", default=None)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--max-candidates", type=int, default=100)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--plot", default=None, help="optional PNG path for target/top-candidate curves")
    _add_xrd_options(evaluate, metadata_defaults=True)
    evaluate.set_defaults(func=evaluate_candidates)

    convert = subparsers.add_parser(
        "convert-samples", help="convert a streamed CrystalFormer sample CSV to CIFs"
    )
    convert.add_argument("--input", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--max-candidates", type=int, default=100)
    convert.add_argument("--require-formula-match", action="store_true")
    convert.set_defaults(func=convert_samples)

    baseline = subparsers.add_parser(
        "random-move", help="run a bounded no-prior random-move Monte Carlo baseline"
    )
    baseline.add_argument("--target", required=True, help="target CSV pattern or target CIF")
    baseline.add_argument("--formula", required=True)
    baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--ground-truth", default=None)
    baseline.add_argument("--evaluations", type=int, default=500)
    baseline.add_argument("--restarts", type=int, default=4)
    baseline.add_argument("--initial-temperature", type=float, default=0.05)
    baseline.add_argument("--final-temperature", type=float, default=0.002)
    baseline.add_argument("--coordinate-sigma", type=float, default=0.04)
    baseline.add_argument("--lattice-sigma", type=float, default=0.04)
    baseline.add_argument("--min-distance", type=float, default=0.65)
    baseline.add_argument("--volume-per-atom-min", type=float, default=10.0)
    baseline.add_argument("--volume-per-atom-max", type=float, default=35.0)
    baseline.add_argument("--length-min", type=float, default=2.0)
    baseline.add_argument("--length-max", type=float, default=20.0)
    baseline.add_argument("--angle-min", type=float, default=55.0)
    baseline.add_argument("--angle-max", type=float, default=125.0)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--plot", default=None)
    _add_xrd_options(baseline, metadata_defaults=True)
    baseline.set_defaults(func=run_baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
