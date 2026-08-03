"""CPU-side powder XRD scoring utilities for CrystalFormer.

The XRD calculator is deliberately kept outside JAX transformations.  Crystal
Former supplies a batch of ``(G, L, XYZ, A, W)`` samples, this module expands
one sample to a pymatgen structure, evaluates the black-box simulator, and
returns a scalar peak-matching score.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Pymatgen's diffraction imports matplotlib indirectly.  A constrained WSL
# checkout often has a read-only home config directory, so keep its tiny cache
# in a writable temporary location unless the caller supplied a preference.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/crystalformer-mpl")

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

_SYMMETRY_OPS: np.ndarray | None = None
_MULTIPLICITIES: np.ndarray | None = None
_WMAX: np.ndarray | None = None


def _symmetry_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the large Wyckoff tables only when a generated sample is scored."""

    global _SYMMETRY_OPS, _MULTIPLICITIES, _WMAX
    if _SYMMETRY_OPS is None:
        from crystalformer.src.wyckoff import mult_table, symops, wmax_table

        _SYMMETRY_OPS = np.asarray(symops)
        _MULTIPLICITIES = np.asarray(mult_table)
        _WMAX = np.asarray(wmax_table)
    return _SYMMETRY_OPS, _MULTIPLICITIES, _WMAX


@dataclass(frozen=True)
class XRDConfig:
    """Configuration shared by target preprocessing and candidate scoring."""

    wavelength: str | float = "CuKa"
    two_theta_min: float = 5.0
    two_theta_max: float = 90.0
    grid_step: float = 0.05
    profile: str = "gaussian"
    fwhm: float = 0.10
    eta: float = 0.5
    target_is_peaks: bool = False
    peak_min_height: float = 0.08
    peak_min_prominence: float = 0.05
    peak_min_distance: float = 0.15
    peak_smoothing: float = 0.10
    peak_min_width: float = 0.05
    peak_max_count: int = 40
    peak_q_tolerance: float = 0.04
    peak_scale_min: float = 0.70
    peak_scale_max: float = 1.40
    peak_zero_shift: float = 0.03

    def __post_init__(self) -> None:
        if not np.isfinite(self.two_theta_min) or not np.isfinite(self.two_theta_max):
            raise ValueError("two-theta bounds must be finite")
        if self.two_theta_min < 0.0 or self.two_theta_max > 180.0:
            raise ValueError("two-theta bounds must lie in [0, 180]")
        if self.two_theta_max <= self.two_theta_min:
            raise ValueError("two_theta_max must be greater than two_theta_min")
        if not np.isfinite(self.grid_step) or self.grid_step <= 0:
            raise ValueError("grid_step must be positive")
        if not np.isfinite(self.fwhm) or self.fwhm <= 0:
            raise ValueError("fwhm must be positive")
        if self.profile not in {"gaussian", "pseudo-voigt", "pvoigt"}:
            raise ValueError("profile must be 'gaussian' or 'pseudo-voigt'")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if isinstance(self.wavelength, (int, float)) and not np.isfinite(self.wavelength):
            raise ValueError("numeric wavelength must be finite")
        for name in ("peak_min_height", "peak_min_prominence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.peak_min_distance <= 0 or self.peak_q_tolerance <= 0:
            raise ValueError("peak distance and q tolerance must be positive")
        if self.peak_smoothing < 0 or self.peak_min_width < 0:
            raise ValueError("peak smoothing and width must be non-negative")
        if self.peak_max_count <= 0:
            raise ValueError("peak_max_count must be positive")
        if not 0 < self.peak_scale_min <= 1.0 <= self.peak_scale_max:
            raise ValueError("peak scale range must contain 1.0")
        if self.peak_zero_shift < 0:
            raise ValueError("peak_zero_shift must be non-negative")

    @property
    def two_theta_range(self) -> tuple[float, float]:
        return (self.two_theta_min, self.two_theta_max)


def make_two_theta_grid(
    two_theta_range: Sequence[float], grid_step: float
) -> np.ndarray:
    """Create a stable, inclusive 2-theta grid."""

    start, stop = (float(two_theta_range[0]), float(two_theta_range[1]))
    if stop <= start:
        raise ValueError("two_theta_range must be increasing")
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("grid_step must be positive")
    count = int(math.floor((stop - start) / grid_step + 1e-10))
    if count > 1_000_000:
        raise ValueError("two-theta grid is too large; increase grid_step")
    grid = start + np.arange(count + 1, dtype=np.float64) * grid_step
    if grid[-1] < stop - 1e-9:
        grid = np.append(grid, stop)
    else:
        grid[-1] = stop
    return grid


def _as_peak_arrays(
    positions: Iterable[Any], intensities: Iterable[Any]
) -> tuple[np.ndarray, np.ndarray]:
    # ``genfromtxt`` returns 0-D arrays for a one-row named CSV.  Converting
    # through ``list`` would then raise ``TypeError``; scalar-safe conversion
    # also keeps mappings and pandas-like series on the same path.
    def as_vector(values: Iterable[Any]) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            array = np.asarray(list(values), dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1)
        return array.reshape(-1)

    positions = as_vector(positions)
    intensities = as_vector(intensities)
    if positions.size != intensities.size:
        raise ValueError("peak positions and intensities must have the same length")
    finite = np.isfinite(positions) & np.isfinite(intensities)
    positions = positions[finite]
    intensities = np.maximum(intensities[finite], 0.0)
    return positions, intensities


def broaden_peaks(
    positions: Iterable[Any],
    intensities: Iterable[Any],
    grid: Sequence[float],
    *,
    profile: str = "gaussian",
    fwhm: float = 0.10,
    eta: float = 0.5,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Convolve discrete peaks with a Gaussian or pseudo-Voigt profile."""

    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    if grid.size == 0:
        return np.empty(0, dtype=np.float64)
    if not np.isfinite(grid).all() or (grid.size > 1 and np.any(np.diff(grid) < 0)):
        raise ValueError("grid must contain finite, non-decreasing values")
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    if profile not in {"gaussian", "pseudo-voigt", "pvoigt"}:
        raise ValueError("profile must be 'gaussian' or 'pseudo-voigt'")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    positions, intensities = _as_peak_arrays(positions, intensities)
    if positions.size == 0 or np.all(intensities <= 0):
        return np.zeros_like(grid)

    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gamma = fwhm / 2.0
    # A measured pattern can contain many more rows than a simulated peak
    # list.  Process peaks in bounded chunks so a malformed/large input cannot
    # allocate a grid-by-all-peaks matrix that exhausts a small WSL host.
    curve = np.zeros_like(grid)
    # Bound the temporary ``grid x peak_chunk`` matrices by roughly 2 million
    # elements even when a user supplies a very fine measured grid.
    bounded_chunk = max(1, min(int(chunk_size), 2_000_000 // max(grid.size, 1)))
    for start in range(0, positions.size, bounded_chunk):
        stop = min(start + bounded_chunk, positions.size)
        delta = grid[:, None] - positions[start:stop][None, :]
        gaussian = np.exp(-0.5 * (delta / sigma) ** 2)
        if profile == "gaussian":
            profile_values = gaussian
        else:
            lorentzian = 1.0 / (1.0 + (delta / gamma) ** 2)
            profile_values = (1.0 - eta) * gaussian + eta * lorentzian
        curve += profile_values @ intensities[start:stop]
    curve[~np.isfinite(curve)] = 0.0
    return np.maximum(curve, 0.0)


def cosine_similarity(
    predicted: Sequence[float], target: Sequence[float], *, clip: bool = True
) -> float:
    """Return a finite cosine similarity for two non-negative curves."""

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.size != target.size:
        raise ValueError("predicted and target curves must have the same length")
    if predicted.size == 0:
        return 0.0
    predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    predicted = np.maximum(predicted, 0.0)
    target = np.maximum(target, 0.0)
    denom = np.linalg.norm(predicted) * np.linalg.norm(target)
    if denom <= np.finfo(np.float64).eps:
        return 0.0
    value = float(np.dot(predicted, target) / denom)
    if np.isclose(value, 1.0, rtol=0.0, atol=1e-12):
        value = 1.0
    return float(np.clip(value, 0.0, 1.0)) if clip else value


@dataclass(frozen=True)
class PeakMatchResult:
    """Peak-level score and the nuisance correction that produced it."""

    score: float
    scale: float
    zero_shift: float
    matched_peaks: int
    target_peaks: int
    candidate_peaks: int


def extract_pattern_peaks(
    grid: Sequence[float],
    curve: Sequence[float],
    config: XRDConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a bounded set of robust local maxima from a sampled pattern."""

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths

    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    if grid.size != curve.size:
        raise ValueError("grid and curve must have the same length")
    if grid.size < 3 or not np.any(curve > 0):
        return np.empty(0), np.empty(0)
    curve = np.maximum(np.nan_to_num(curve), 0.0)
    step = float(np.median(np.diff(grid)))
    detection_curve = curve
    if config.peak_smoothing > 0:
        sigma = max(config.peak_smoothing / max(step, 1e-12), 0.01)
        detection_curve = gaussian_filter1d(
            curve, sigma=sigma, mode="nearest", truncate=4.0
        )
    maximum = float(np.max(detection_curve))
    minimum_distance = max(1, int(round(config.peak_min_distance / step)))
    indices, properties = find_peaks(
        detection_curve,
        height=config.peak_min_height * maximum,
        prominence=config.peak_min_prominence * maximum,
        distance=minimum_distance,
    )
    if indices.size == 0:
        return np.empty(0), np.empty(0)
    intensities = np.asarray(properties["peak_heights"], dtype=np.float64)
    measured_widths = peak_widths(detection_curve, indices, rel_height=0.5)[0] * step
    smoothing_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * config.peak_smoothing
    intrinsic_widths = np.sqrt(np.maximum(measured_widths**2 - smoothing_fwhm**2, 0.0))
    keep = intrinsic_widths >= config.peak_min_width
    indices, intensities = indices[keep], intensities[keep]
    if indices.size == 0:
        return np.empty(0), np.empty(0)
    if indices.size > config.peak_max_count:
        keep = np.argsort(intensities)[-config.peak_max_count:]
        indices, intensities = indices[keep], intensities[keep]
    order = np.argsort(indices)
    intensities = intensities[order] / max(float(np.max(intensities)), 1e-12)
    return grid[indices[order]], intensities


def _two_theta_to_q(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
    theta = np.deg2rad(np.asarray(two_theta, dtype=np.float64) / 2.0)
    return 4.0 * np.pi * np.sin(theta) / wavelength


def _greedy_peak_score(
    target_q: np.ndarray,
    target_weights: np.ndarray,
    candidate_q: np.ndarray,
    candidate_weights: np.ndarray,
    tolerance: float,
) -> tuple[float, int]:
    """Match sorted peaks one-to-one and return a position-weighted F1."""

    i = j = matched = 0
    match_weight = 0.0
    while i < target_q.size and j < candidate_q.size:
        delta = candidate_q[j] - target_q[i]
        if delta < -tolerance:
            j += 1
            continue
        if delta > tolerance:
            i += 1
            continue
        # Select the closer of the current and next candidate peak.
        if j + 1 < candidate_q.size:
            next_delta = candidate_q[j + 1] - target_q[i]
            if abs(next_delta) < abs(delta) and abs(next_delta) <= tolerance:
                j += 1
                delta = next_delta
        quality = math.exp(-0.5 * (delta / tolerance) ** 2)
        match_weight += math.sqrt(target_weights[i] * candidate_weights[j]) * quality
        matched += 1
        i += 1
        j += 1
    precision = match_weight / max(float(np.sum(candidate_weights)), 1e-12)
    recall = match_weight / max(float(np.sum(target_weights)), 1e-12)
    if precision + recall <= 0:
        return 0.0, 0
    return float(2.0 * precision * recall / (precision + recall)), matched


def peak_match_similarity(
    predicted: Sequence[float],
    target: Sequence[float],
    grid: Sequence[float],
    config: XRDConfig,
) -> PeakMatchResult:
    """Compare peak sets after correcting global lattice scale and zero shift."""

    grid = np.asarray(grid, dtype=np.float64)
    target_theta, target_intensity = extract_pattern_peaks(grid, target, config)
    candidate_theta, candidate_intensity = extract_pattern_peaks(grid, predicted, config)
    if target_theta.size == 0 or candidate_theta.size == 0:
        return PeakMatchResult(0.0, 1.0, 0.0, 0, target_theta.size, candidate_theta.size)

    wavelength = float(XRDCalculator(wavelength=config.wavelength).wavelength)
    target_q = _two_theta_to_q(target_theta, wavelength)
    candidate_q = _two_theta_to_q(candidate_theta, wavelength)
    # Weak peaks remain useful evidence without carrying nearly the same weight
    # as a strong, repeatable peak.
    target_weights = np.sqrt(target_intensity)
    candidate_weights = np.sqrt(candidate_intensity)

    strongest_target = np.argsort(target_intensity)[-min(12, target_q.size):]
    strongest_candidate = np.argsort(candidate_intensity)[-min(12, candidate_q.size):]
    hypotheses = [1.0]
    for ti in strongest_target:
        for ci in strongest_candidate:
            scale = target_q[ti] / candidate_q[ci]
            if config.peak_scale_min <= scale <= config.peak_scale_max:
                hypotheses.append(float(scale))

    best = (0.0, 1.0, 0.0, 0)
    coarse_offsets = np.linspace(
        -config.peak_zero_shift, config.peak_zero_shift, 5
    )
    for scale in hypotheses:
        for offset in coarse_offsets:
            transformed = scale * candidate_q + offset
            score, matched = _greedy_peak_score(
                target_q, target_weights, transformed, candidate_weights,
                config.peak_q_tolerance,
            )
            if score > best[0]:
                best = (score, scale, float(offset), matched)

    # A small local refinement avoids making the answer depend on coarse guesses.
    _, best_scale, best_offset, _ = best
    for scale in np.linspace(
        max(config.peak_scale_min, best_scale - 0.01),
        min(config.peak_scale_max, best_scale + 0.01),
        21,
    ):
        for offset in np.linspace(
            max(-config.peak_zero_shift, best_offset - 0.01),
            min(config.peak_zero_shift, best_offset + 0.01),
            9,
        ):
            transformed = scale * candidate_q + offset
            score, matched = _greedy_peak_score(
                target_q, target_weights, transformed, candidate_weights,
                config.peak_q_tolerance,
            )
            if score > best[0]:
                best = (score, float(scale), float(offset), matched)

    score, scale, offset, matched = best
    return PeakMatchResult(
        float(np.clip(score, 0.0, 1.0)), scale, offset, matched,
        int(target_q.size), int(candidate_q.size),
    )


def _symmetrize_atoms(g: Any, w: Any, x: Sequence[float]) -> np.ndarray:
    """Expand one sampled Wyckoff representative to fractional coordinates."""

    symmetry_ops, multiplicities, wmax = _symmetry_tables()
    g = int(np.asarray(g))
    w = int(np.asarray(w))
    x = np.asarray(x, dtype=np.float64).reshape(3)
    if not 1 <= g <= 230:
        raise ValueError(f"space group must be in [1, 230], got {g}")
    if not 1 <= w < multiplicities.shape[1]:
        raise ValueError(f"invalid Wyckoff index: {w}")

    w_max = int(wmax[g - 1])
    m_max = int(multiplicities[g - 1, w_max])
    ops = symmetry_ops[g - 1, w_max, :m_max]
    affine_point = np.concatenate((x, [1.0]))
    coords = ops @ affine_point
    coords -= np.floor(coords)

    # The sampled point may correspond to a different orbit representative.
    def distance_to_first_op(coord: np.ndarray) -> float:
        diff = symmetry_ops[g - 1, w, 0] @ np.concatenate((coord, [1.0]))
        diff = diff - coord
        diff -= np.rint(diff)
        return float(np.sum(diff**2))

    x = coords[int(np.argmin([distance_to_first_op(coord) for coord in coords]))]
    multiplicity = int(multiplicities[g - 1, w])
    ops = symmetry_ops[g - 1, w, :multiplicity]
    coords = ops @ np.concatenate((x, [1.0]))
    coords -= np.floor(coords)
    return np.asarray(coords, dtype=np.float64)


def structure_from_GLXYZAW(
    G: Any, L: Sequence[float], XYZ: Sequence[Sequence[float]],
    A: Sequence[int], W: Sequence[int]
) -> Structure:
    """Convert CrystalFormer's padded Wyckoff representation to Structure."""

    g = int(np.asarray(G))
    lattice_parameters = np.asarray(L, dtype=np.float64).reshape(-1)
    xyz = np.asarray(XYZ, dtype=np.float64)
    atoms = np.asarray(A).reshape(-1)
    wyckoff = np.asarray(W).reshape(-1)
    if lattice_parameters.size != 6:
        raise ValueError("L must contain a, b, c, alpha, beta, gamma")
    if not np.isfinite(lattice_parameters).all():
        raise ValueError("lattice parameters must be finite")
    if np.any(lattice_parameters[:3] < 1e-4) or np.any(lattice_parameters[:3] > 1e4):
        raise ValueError("lattice lengths are outside the safe physical range")
    if np.any(lattice_parameters[3:] <= 0) or np.any(lattice_parameters[3:] >= 180):
        raise ValueError("lattice angles must lie strictly between 0 and 180 degrees")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("XYZ must have shape (n_sites, 3)")
    if not (xyz.shape[0] == atoms.size == wyckoff.size):
        raise ValueError("XYZ, A and W must have the same number of sites")

    active = atoms > 0
    if not np.any(active):
        raise ValueError("sample contains no active atomic sites")
    if not np.isfinite(xyz[active]).all():
        raise ValueError("active fractional coordinates must be finite")
    lattice = Lattice.from_parameters(*lattice_parameters)
    if not np.isfinite(lattice.volume) or lattice.volume < 1e-6:
        raise ValueError("lattice volume is too small or non-finite")
    species: list[int] = []
    coordinates: list[np.ndarray] = []
    for atom, position, wp in zip(atoms[active], xyz[active], wyckoff[active]):
        orbit = _symmetrize_atoms(g, wp, position)
        species.extend([int(atom)] * len(orbit))
        coordinates.extend(orbit)
    if not coordinates:
        raise ValueError("sample expanded to no atomic coordinates")
    structure = Structure(lattice, species, np.asarray(coordinates), coords_are_cartesian=False)
    return structure.get_primitive_structure()


def _load_two_column_pattern(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV/TSV/whitespace pattern with two numeric columns."""

    delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        named = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float)
        if named.dtype.names:
            def normalized_name(name: str) -> str:
                # Accommodate common instrument headers such as
                # ``2theta (deg)`` and a UTF-8 BOM without depending on a
                # particular vendor's spelling.
                name = name.lstrip("\ufeff").strip().lower()
                name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
                return name

            names = {
                normalized_name(name): name
                for name in named.dtype.names
            }
            x_name = next(
                (names[key] for key in (
                    "two_theta", "2theta", "twotheta", "two_theta_deg",
                    "2theta_deg", "angle", "angle_deg", "theta", "x",
                ) if key in names),
                None,
            )
            y_name = next(
                (names[key] for key in (
                    "intensity", "intensities", "intensity_counts",
                    "intensity_au", "counts", "count", "y",
                ) if key in names),
                None,
            )
            if x_name is not None and y_name is not None:
                return _as_peak_arrays(named[x_name], named[y_name])

            # A named file may use arbitrary vendor labels.  Use its first
            # two columns when the header is clearly textual; retain the
            # numeric-header case for the fallback below so a headerless file
            # does not lose its first data row.
            def is_numeric_name(name: str) -> bool:
                try:
                    float(name)
                    return True
                except (TypeError, ValueError):
                    return False

            if len(named.dtype.names) >= 2 and not all(
                is_numeric_name(name) for name in named.dtype.names[:2]
            ):
                return _as_peak_arrays(
                    named[named.dtype.names[0]], named[named.dtype.names[1]]
                )
    except (OSError, ValueError):
        pass

    data = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"pattern file must contain at least two columns: {path}")
    return data[:, 0], data[:, 1]


def load_xrd_pattern(target: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize common pattern representations to ``(two_theta, intensity)``."""

    if isinstance(target, (str, Path)):
        path = Path(target)
        if path.suffix.lower() in {".cif", ".json"}:
            raise ValueError("a structure file is not a pattern; pass it as target_structure")
        return _load_two_column_pattern(path)

    if isinstance(target, Mapping):
        x = target.get("two_theta", target.get("theta", target.get("x")))
        y = target.get("intensity", target.get("intensities", target.get("y")))
        if x is None or y is None:
            raise ValueError("pattern mapping needs two_theta/theta/x and intensity/y")
        return _as_peak_arrays(x, y)

    if hasattr(target, "x") and hasattr(target, "y"):
        return _as_peak_arrays(target.x, target.y)

    if isinstance(target, (tuple, list)) and len(target) == 2:
        return _as_peak_arrays(target[0], target[1])

    array = np.asarray(target, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 2 and array.shape[1] != 2:
        # Accept the common ``np.vstack((two_theta, intensity))`` form too.
        return _as_peak_arrays(array[0], array[1])
    if array.ndim == 2 and array.shape[1] >= 2:
        return _as_peak_arrays(array[:, 0], array[:, 1])
    if array.ndim == 2 and array.shape[0] == 2:
        return _as_peak_arrays(array[0], array[1])
    raise TypeError("unsupported XRD target representation")


def simulate_structure_pattern(
    structure: Structure,
    calculator: XRDCalculator,
    config: XRDConfig,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate and broaden a structure's powder pattern on a fixed grid."""

    if grid is None:
        grid = make_two_theta_grid(config.two_theta_range, config.grid_step)
    pattern = calculator.get_pattern(
        structure,
        two_theta_range=config.two_theta_range,
        scaled=True,
    )
    curve = broaden_peaks(
        pattern.x,
        pattern.y,
        grid,
        profile=config.profile,
        fwhm=config.fwhm,
        eta=config.eta,
    )
    return grid, curve


def _target_curve(
    target: Any,
    target_intensity: Any,
    calculator: XRDCalculator,
    config: XRDConfig,
    grid: np.ndarray,
    target_structure: Structure | str | Path | None,
) -> np.ndarray:
    if target_structure is not None:
        if isinstance(target_structure, (str, Path)):
            target_structure = Structure.from_file(str(target_structure))
        _, curve = simulate_structure_pattern(target_structure, calculator, config, grid)
        return curve

    if target is None:
        raise ValueError("target, target_pattern, or target_structure is required")
    if isinstance(target, Structure):
        _, curve = simulate_structure_pattern(target, calculator, config, grid)
        return curve
    if isinstance(target, (str, Path)) and Path(target).suffix.lower() in {".cif", ".json"}:
        structure = Structure.from_file(str(target))
        _, curve = simulate_structure_pattern(structure, calculator, config, grid)
        return curve

    if target_intensity is not None:
        positions, intensities = _as_peak_arrays(target, target_intensity)
    else:
        positions, intensities = load_xrd_pattern(target)
    if config.target_is_peaks:
        return broaden_peaks(
            positions,
            intensities,
            grid,
            profile=config.profile,
            fwhm=config.fwhm,
            eta=config.eta,
        )

    finite = np.isfinite(positions) & np.isfinite(intensities)
    positions = positions[finite]
    intensities = np.maximum(intensities[finite], 0.0)
    if positions.size == 0:
        return np.zeros_like(grid)
    order = np.argsort(positions)
    positions = positions[order]
    intensities = intensities[order]
    # np.interp accepts duplicate x values but its choice of the duplicate is
    # implementation-dependent.  Aggregate them deterministically.
    unique_positions, inverse = np.unique(positions, return_inverse=True)
    if unique_positions.size != positions.size:
        sums = np.zeros(unique_positions.size, dtype=np.float64)
        counts = np.zeros(unique_positions.size, dtype=np.int64)
        np.add.at(sums, inverse, intensities)
        np.add.at(counts, inverse, 1)
        intensities = sums / np.maximum(counts, 1)
        positions = unique_positions
    return np.interp(grid, positions, intensities, left=0.0, right=0.0)


def make_xrd_reward_fn(
    target: Any = None,
    target_intensity: Any = None,
    *,
    target_pattern: Any = None,
    target_structure: Structure | str | Path | None = None,
    wavelength: str | float = "CuKa",
    two_theta_range: Sequence[float] = (5.0, 90.0),
    grid_step: float = 0.05,
    profile: str = "gaussian",
    fwhm: float = 0.10,
    eta: float = 0.5,
    target_is_peaks: bool = False,
    peak_smoothing: float = 0.10,
    peak_min_width: float = 0.05,
    invalid_reward: float = 0.0,
    output_name: str = "xrd_scores",
):
    """Build a pair of scalar and batched peak-matching reward functions.

    The scalar reward is a peak precision/recall score in ``[0, 1]``.  The batched
    function accepts the same optional ``path`` and ``epoch`` arguments as the
    existing conditional PPO rewards and executes entirely on the host.
    """

    if target is not None and target_pattern is not None:
        raise ValueError("pass only one of target and target_pattern")
    if (target is not None or target_pattern is not None) and target_structure is not None:
        raise ValueError("pass either target/target_pattern or target_structure, not both")
    if target is None:
        target = target_pattern
    if isinstance(wavelength, str):
        try:
            wavelength = float(wavelength)
        except ValueError:
            pass
    if not np.isfinite(invalid_reward):
        raise ValueError("invalid_reward must be finite")
    config = XRDConfig(
        wavelength=wavelength,
        two_theta_min=float(two_theta_range[0]),
        two_theta_max=float(two_theta_range[1]),
        grid_step=float(grid_step),
        profile=profile,
        fwhm=float(fwhm),
        eta=float(eta),
        target_is_peaks=bool(target_is_peaks),
        peak_smoothing=float(peak_smoothing),
        peak_min_width=float(peak_min_width),
    )
    grid = make_two_theta_grid(config.two_theta_range, config.grid_step)
    calculator = XRDCalculator(wavelength=config.wavelength)
    target_curve = _target_curve(
        target,
        target_intensity,
        calculator,
        config,
        grid,
        target_structure,
    )
    target_curve = np.maximum(
        np.nan_to_num(target_curve, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    if not np.any(target_curve > 0):
        raise ValueError("target XRD pattern has no positive intensity on the configured grid")

    def _score_structure(structure: Structure) -> float:
        _, curve = simulate_structure_pattern(structure, calculator, config, grid)
        return peak_match_similarity(curve, target_curve, grid, config).score

    def reward_fn(x: Sequence[Any]) -> float:
        try:
            return _score_structure(structure_from_GLXYZAW(*x))
        except Exception:
            return float(invalid_reward)

    def batch_reward_fn(x: Sequence[Any], path: str | None = None, epoch: int | None = None):
        host_x = tuple(np.asarray(value) for value in x)
        if len(host_x) != 5 or any(value.ndim == 0 for value in host_x):
            raise ValueError("XRD batch input must be five arrays with a leading sample dimension")
        sample_sizes = {value.shape[0] for value in host_x}
        if len(sample_sizes) != 1:
            raise ValueError("all XRD batch arrays must have the same leading dimension")
        candidate_path = None
        if path is not None and epoch is not None:
            candidate_path = Path(path) / f"{output_name}_{epoch}_candidates"
            candidate_path.mkdir(parents=True, exist_ok=True)
        scores_list = []
        for index, sample in enumerate(zip(*host_x)):
            try:
                structure = structure_from_GLXYZAW(*sample)
                if candidate_path is not None:
                    # Persist the converted candidate before simulation.  A
                    # simulator failure should still leave a CIF for
                    # post-mortem validation and StructureMatcher analysis.
                    structure.to(
                        fmt="cif",
                        filename=str(candidate_path / f"sample_{index:04d}.cif"),
                    )
                scores_list.append(_score_structure(structure))
            except Exception:
                scores_list.append(float(invalid_reward))
        scores = np.asarray(scores_list, dtype=np.float32)
        if path is not None and epoch is not None:
            output_path = Path(path)
            output_path.mkdir(parents=True, exist_ok=True)
            with (output_path / f"{output_name}_{epoch}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("sample", "xrd_similarity"))
                writer.writerows(enumerate(scores.tolist()))
        # Returning a NumPy vector keeps the simulator independent of JAX.  The
        # PPO loop converts it to a JAX array after the host-side callback.
        return scores

    # Small introspection hooks make target metadata reusable by evaluation code.
    batch_reward_fn.grid = grid
    batch_reward_fn.target_curve = target_curve
    batch_reward_fn.config = config
    return reward_fn, batch_reward_fn


def save_pattern(
    path: str | Path,
    grid: Sequence[float],
    intensity: Sequence[float],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a reproducible two-column pattern and optional JSON metadata."""

    path = Path(path)
    grid = np.asarray(grid).reshape(-1)
    intensity = np.asarray(intensity).reshape(-1)
    if grid.size != intensity.size:
        raise ValueError("grid and intensity must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("two_theta", "intensity"))
        writer.writerows(zip(grid.tolist(), intensity.tolist()))
    if metadata is not None:
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata_path.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True))


def config_dict(config: XRDConfig) -> dict[str, Any]:
    """Return JSON-friendly configuration metadata."""

    return asdict(config)
