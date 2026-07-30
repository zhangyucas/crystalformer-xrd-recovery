# Issue #68 Powder XRD Recovery Study

## Status

The Issue #68 workflow is implemented end to end: composition-conditioned
CrystalFormer sampling, host-side powder-XRD cosine reward, PPO with a fixed
pretrained prior, tau and space-group exploration controls, random-move Monte
Carlo baseline, candidate conversion, StructureMatcher evaluation, plots, and
low-memory GPU execution. The experiments below were run on 2026-07-29.

## Hardware-safe protocol

- GPU: RTX 4060 Laptop, 8188 MiB VRAM.
- WSL: 7.6 GiB RAM and 2.0 GiB swap.
- Full model: 16 layers, width 256, 13,836,461 parameters.
- Sampling: GPU batch 1 with a params-only 52.8 MiB checkpoint.
- PPO: batch 8, sampling batch 1, gradient microbatch 2, serial XRD scoring.
- Guards: no XLA preallocation, one BLAS thread, serialized compilation, and
  termination below 1.5 GiB available RAM or 0.5 GiB free swap.

The largest completed PPO run used 2.75 GiB process RSS and left at least
2.15 GiB WSL memory available. Increasing the WSL limit was therefore not
required; changing `.wslconfig` would require `wsl --shutdown` and was not done
during the live run.

## Benchmarks

All targets are clean self-supervised patterns simulated with Cu K-alpha over
5-90 degrees. NaCl, MgO, and KCl use a 0.05 degree grid and Gaussian FWHM 0.5
degree. The earlier simple-Si target uses a 0.2 degree grid and FWHM 0.2 degree.
Only candidates with the requested reduced formula are evaluated.

| target | method | requested / evaluated | mean cosine | best cosine | matched candidates | top-1 match |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| simple Si | prior | 40 / 40 | 0.141185 | 0.450072 | 0 / 40 | no |
| simple Si | PPO, batch 8 | 40 / 40 | 0.149443 | 0.504783 | 0 / 40 | no |
| simple Si | random move | 40 / 40 | - | 0.097692 | 0 / 1 best | no |
| NaCl rock salt | prior | 80 / 80 | 0.138215 | 0.469789 | 79 / 80 | yes |
| NaCl rock salt | PPO, batch 8 | 80 / 80 | 0.181079 | 0.571371 | 79 / 80 | yes |
| NaCl rock salt | random move | 80 / 80 | - | 0.656109 | 0 / 1 best | no |
| MgO rock salt | prior | 80 / 9 | 0.051633 | 0.142385 | 0 / 9 | no |
| MgO rock salt | random move | 80 / 78 | - | 0.307052 | 0 / 1 best | no |
| KCl rock salt | prior | 80 / 15 | 0.000050 | 0.000160 | 0 / 15 | no |
| KCl rock salt | random move | 80 / 80 | - | 0.327658 | 0 / 1 best | no |

`requested / evaluated` exposes formula failures rather than silently dropping
them. MgO produced only 9 exact 1:1 candidates; KCl produced 15. Candidate
match fraction is not a recovery rate over independent targets. By the strict
top-1-per-target metric, this small four-target study recovered 1/4 targets.

## PPO result

The strongest positive run is NaCl with five epochs, 40 training candidates,
`tau=0.1`, `lr=5e-7`, and fresh Adam state. Independent paired evaluation uses
the same 80 sampling keys before and after training:

- 79/80 paired candidates improved.
- Mean cosine increased by 0.042863, from 0.138215 to 0.181079.
- Best cosine increased by 0.101582, from 0.469789 to 0.571371.
- Structure matches stayed at 79/80 and the top-ranked candidate stayed correct.
- Training took 60.66 seconds, peaked at 2.73 GiB RSS, and did not trigger the guard.

The random-move NaCl baseline reached a larger raw cosine of 0.656109 but was
not structure-matched. This is the intended distinction between a spectral
look-alike and a recovered structure: the prior supplies the correct rock-salt
topology, while PPO moves its lattice distribution toward the target.

Plots:

- [NaCl prior patterns](../experiments/gpu_20260729/runs/prior/base_nacl_seed1_n80/top5_patterns_fwhm05.png)
- [NaCl PPO patterns](../experiments/gpu_20260729/runs/ppo/nacl_microbatch_b8_tau01_seed6_eval_seed1_n80/top5_patterns.png)
- [NaCl random-move best](../experiments/gpu_20260729/runs/baselines/random_move_nacl_seed1_n80_fwhm05/best_pattern.png)

## Tau ablation

The simple-Si ablation used three epochs, batch 2, `lr=5e-7`, and 40 paired
evaluation samples. It improved XRD fit but did not recover the structure.

| tau | mean | median | best | improved vs prior | mean delta | matches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prior | 0.141185 | 0.114091 | 0.450072 | - | - | 0 / 40 |
| 0.0 | 0.151253 | 0.126321 | 0.482996 | 36 / 40 | +0.010069 | 0 / 40 |
| 0.1 | 0.149401 | 0.124713 | 0.476006 | 36 / 40 | +0.008216 | 0 / 40 |
| 0.5 | 0.142721 | 0.115434 | 0.445904 | 24 / 40 | +0.001537 | 0 / 40 |

The larger `tau` is more conservative over this short run. These data do not
show that tau alone cures local minima because every Si candidate is still a
StructureMatcher failure.

## Failure analysis

- Simple Si: all default samples selected space group 12. The target's
  symmetry-analyzed group is 221, whose prior probability is 2.14e-6 at
  space-group temperature 1 and 0.01176 at temperature 5. Forced group 221
  sampling still gave 0/80 matches, so symmetry exploration is not the only
  problem for this out-of-prior simple-cubic Si target.
- MgO: 80 samples contained 19 group-225 proposals, but none preserved exact
  Mg:O=1:1. Formula rejection leaves too few candidates for bounded PPO.
- KCl: 72/80 samples emitted group 225, but the 15 exact-formula candidates
  reduced by symmetry analysis to the CsCl-type group 221 topology.
- Profile sensitivity: for the same NaCl candidates, target FWHM 0.1, 0.2,
  0.5, and 1.0 gave best cosine 0.004540, 0.024750, 0.469789, and 0.802847.
  Peak broadening is therefore part of the inverse model, not a cosmetic plot
  parameter. FWHM 0.5 was retained as the balanced setting.

## Verification and limits

- Focused XRD/PPO tests: 27 passed.
- Low-memory repository regression: 61 passed, with
  `tests/test_transformer.py` excluded because its full Jacobian test has a
  known high host-memory peak on this WSL allocation.
- This is a four-target self-supervised study, not a SimXRD-4M recovery-rate
  claim and not an experimental opXRD result.
- No structure relaxation or DFT is involved; StructureMatcher validates
  identity only against the known benchmark CIF.

Machine-readable metrics are in
[final_metrics.csv](../experiments/gpu_20260729/metrics/final_metrics.csv).
