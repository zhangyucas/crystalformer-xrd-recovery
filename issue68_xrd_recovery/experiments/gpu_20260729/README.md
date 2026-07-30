# GPU experiment artifacts: 2026-07-29

This directory preserves the complete Issue #68 experiment campaign executed
on an RTX 4060 Laptop GPU under a 7.6 GiB WSL allocation.

```text
gpu_20260729/
|-- README.md
|-- metrics/       # Final cross-method table
|-- targets/       # Simulated XRD targets and metadata
|-- checkpoints/   # Prior and trained params-only checkpoints
|-- cache/         # JAX compilation caches retained for reproducibility
`-- runs/
    |-- prior/       # Pretrained-prior sampling and evaluation
    |-- ppo/         # Completed PPO runs, guards, and paired evaluations
    |-- baselines/   # Random-move Monte Carlo references
    |-- ablations/   # Tau sweep
    `-- diagnostics/ # Probes, stopped runs, and failure analysis
```

Primary evidence:

- `metrics/final_metrics.csv`: final benchmark table.
- `runs/prior/base_nacl_seed1_n80/`: NaCl prior evaluation.
- `runs/ppo/nacl_microbatch_b8_tau01_seed6_eval_seed1_n80/`: paired NaCl
  post-PPO evaluation.
- `runs/ppo/nacl_microbatch_b8_tau01_seed6_guard/guard_summary.json`: PPO
  hardware telemetry summary.
- `runs/baselines/random_move_nacl_seed1_n80_fwhm05/`: no-prior baseline.

Run-produced JSON and CSV files retain their original absolute paths as
provenance. Those strings are historical metadata and are not current entry
points after project organization.
