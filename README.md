# Formula-Aware XRD Crystal Recovery

This repository develops two targeted improvements for formula-conditioned
crystal generation and powder-XRD candidate ranking:

1. **Dynamic-programming composition masking (DP Mask)** removes actions that
   can no longer complete the requested chemical formula.
2. **XRD peak-matching similarity** compares detected diffraction peaks with a
   bounded q-scale and zero-shift correction, instead of comparing complete
   broadened curves only.

The project is intentionally evaluated as two separate problems: chemical
formula validity and XRD ranking. A high XRD score is not treated as proof of
topology, stability, or DFT correctness.

## Results

The completed two-seed benchmark generated 2,400 candidates for 12 real
experimental opXRD targets:

| Metric | Result |
|---|---:|
| Formula matches | 2,400 / 2,400 (100.00%) |
| Geometry precheck/CIF conversion | 2,389 / 2,400 (99.54%) |
| Generated candidates matching the target topology | 48 / 2,400 (2.00%) |

On the identical candidate pools, the new peak score improved the injected
known-structure ranking over the previous whole-curve cosine score:

| Correct structure in | Previous cosine | Peak matching |
|---|---:|---:|
| Top-1 | 9 / 12 | 10 / 12 |
| Top-5 | 9 / 12 | 11 / 12 |
| Top-10 | 9 / 12 | 12 / 12 |

The injected structure is an oracle ranking control, not a generated sample.
The full Chinese report is [here](reports/DP_AND_PEAK_METHODS_REPORT_ZH.md),
and the machine-readable same-pool comparison is
[here](reports/XRD_SCORE_AB_SAME_POOL.csv).

## Method Overview

### Dynamic composition mask

At every space-group, Wyckoff, and element decision, the sampler checks
whether the remaining multiplicities can still reach an integer multiple of
the reduced target formula. Impossible actions are masked before sampling.
The same reachability semantics are used by the formula-conditioned sampling
and replay log-probability paths.

Implementation: `crystalformer/src/composition_reachability.py` and
`crystalformer/src/sample.py`.

### Peak-matching XRD score

The scorer extracts significant peaks, transforms peak positions to q-space,
searches a bounded global scale (`0.70-1.40`) and zero shift (`+-0.03`),
matches peaks one-to-one, and combines weighted precision and recall into one
score in `[0, 1]`. Current detection settings are height `0.08`, prominence
`0.05`, smoothing `0.10` degrees, minimum distance `0.15` degrees, q tolerance
`0.04`, and at most 40 peaks.

Implementation: `crystalformer/reinforce/xrd.py`.

## Quick Start

Install the package in a Python 3.10+ environment:

```bash
pip install .
```

Show the XRD recovery commands:

```bash
python issue68_xrd_recovery/scripts/xrd_recovery.py --help
python issue68_xrd_recovery/scripts/xrd_prior_sample.py --help
```

Evaluate candidate CIFs against a target pattern without loading the model:

```bash
python issue68_xrd_recovery/scripts/xrd_recovery.py evaluate \
  --target TARGET.csv \
  --candidates CANDIDATE_DIR \
  --ground-truth TARGET.cif \
  --output RESULTS.csv
```

Run the focused regression tests:

```bash
python -m pytest -q \
  tests/test_composition_progress_mask.py \
  issue68_xrd_recovery/tests/test_xrd_reward.py
```

## Repository Layout

```text
crystalformer/src/composition_reachability.py  # DP formula reachability
crystalformer/src/sample.py                    # masked conditional sampler
crystalformer/reinforce/xrd.py                 # peak extraction and scoring
issue68_xrd_recovery/scripts/xrd_recovery.py   # target/evaluate utilities
issue68_xrd_recovery/tests/                    # XRD workflow tests
tests/test_composition_progress_mask.py        # DP mask tests
reports/                                        # current report and A/B data
```

Large checkpoints, raw candidate pools, caches, and GPU run directories are
excluded by `.gitignore` and are not required for the lightweight code and
method report.
