# Issue #68: Powder XRD Recovery

This directory contains the project-specific code, tests, reports, and
experiment artifacts for QuantumBFS `quantum.harness` Issue #68. Reusable
CrystalFormer implementation remains under `crystalformer/`, where it belongs
to the installed Python package.

## Directory layout

```text
issue68_xrd_recovery/
|-- README.md                 # Project map and usage
|-- reports/
|   |-- XRD_STUDY.md         # English experiment report
|   `-- XRD_STUDY_ZH.md      # Chinese experiment report
|-- scripts/
|   |-- run_gpu_host.sh      # WSL host-side GPU bridge
|   |-- run_memory_guard.py  # Process and WSL memory guard
|   |-- xrd_prior_sample.py  # Streaming prior sampler
|   |-- xrd_recovery.py      # Target, conversion, evaluation, baseline
|   `-- xrd_study.py         # Tau/seed study command and summary tool
|-- tests/
|   |-- fixtures/            # Known CIF targets
|   `-- test_*.py            # Issue-specific regression tests
`-- experiments/
    `-- gpu_20260729/         # Targets, checkpoints, categorized runs, metrics
```

## Reusable package code

The following files integrate the workflow into CrystalFormer and therefore
remain in the main package instead of being duplicated here:

- `crystalformer/reinforce/xrd.py`: XRD simulation and cosine reward.
- `crystalformer/reinforce/xrd_baseline.py`: random-move baseline.
- `crystalformer/reinforce/ppo.py`: reward direction, replay, diversity, and
  gradient microbatching.
- `crystalformer/cli/train_ppo.py`: production PPO command-line integration.
- `crystalformer/src/sample.py` and `crystalformer/src/transformer.py`:
  sampling exploration controls.

## Entry points

Run commands from the repository root:

```bash
python issue68_xrd_recovery/scripts/xrd_recovery.py --help
python issue68_xrd_recovery/scripts/xrd_study.py --help
python issue68_xrd_recovery/scripts/xrd_prior_sample.py --help
pytest -q issue68_xrd_recovery/tests
```

The experiment JSON and CSV files are immutable run records. Absolute paths
inside them describe their original execution location before this directory
was organized; use paths relative to this directory for current access.

## Results

- [Chinese report](reports/XRD_STUDY_ZH.md)
- [English report](reports/XRD_STUDY.md)
- [Final metrics](experiments/gpu_20260729/metrics/final_metrics.csv)
