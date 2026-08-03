<div align="center">
  <img align="middle" src="imgs/crystalformer.png" width="400" alt="logo"/>
  <h2> A Foundation Model for Crystal Structure Generation and Prediction</h2>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2403.15734-b31b1b.svg)](https://arxiv.org/abs/2403.15734)
[![arXiv](https://img.shields.io/badge/arXiv-2504.02367-b31b1b.svg)](https://arxiv.org/abs/2504.02367)
[![arXiv](https://img.shields.io/badge/arXiv-2512.18251-b31b1b.svg)](https://arxiv.org/abs/2512.18251)

<div align="center">
  <img align="middle" src="imgs/output.gif" width="400">
  <h3> Generating Cs<sub>2</sub>ZnFe(CN)<sub>6</sub> Crystal (<a href=https://next-gen.materialsproject.org/materials/mp-570545>mp-570545</a>) </h3>
</div>

# Overview

_CrystalFormer_ is a unified autoregressive transformer model for inorganic crystalline material generation that supports **both** de novo crystal generation (DNG) and crystal structure prediction (CSP)
within a *single probabilistic framework*.
It is specifically designed for space group-controlled generation of crystalline materials. The space group symmetry significantly simplifies the crystal space, which is crucial for data and compute efficient generative modeling of crystalline materials.

The model can:

- **De novo generation** $p(C|\varnothing)$: Generate plausible crystal structures from scratch, without any formula constraint.
- **Crystal structure prediction** $p(C|f)$: Generate crystal structures conditioned on a given chemical formula $f$.

We provide a pretrained checkpoint to support both functionalities. No architectural change is required — _CrystalFormer_ seamlessly switches behavior depending on whether a formula is supplied. Furthermore, the performance of both functionalities can be boosted via reinforcement fine-tuning.

## Contents

- [Contents](#contents)
- [Model Card](#model-card)
- [Status](#status)
- [Get Started](#get-started)
- [Installation](#installation)
  - [CPU installation](#cpu-installation)
  - [CUDA (GPU) installation](#cuda-gpu-installation)
  - [Install required packages and command line tools](#install-required-packages-and-command-line-tools)
- [Available Weights](#available-weights)
- [Crystal Structure Prediction](#crystal-structure-prediction)
  - [Sample](#sample)
  - [Relax](#relax-generated-structures-with-mlff)
  - [Energy Above Hull (Ehull)](#energy-above-hull-ehull)
  - [End-to-End Pipeline](#end-to-end-pipeline)
  - [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server)
- [De Novo Generation](#de-novo-generation)
  - [Sample](#sample-1)
  - [Evaluate](#evaluate)
- [Advanced Usage](#advanced-usage)
  - [Reinforcement Fine-tuning](#reinforcement-fine-tuning)
  - [Writing custom reward functions](#writing-custom-reward-functions)
  - [Pretrain](#pretrain)
- [How to cite](#how-to-cite)


## Model Card

_CrystalFormer_ is an autoregressive transformer for the probability distribution of crystal structures:

- **De novo generation**: $P(C|\varnothing) = P(g) P(W_1|...) P(A_1|...) P(X_1|...) ... P(L|...)$
- **Formula-conditioned prediction**: $P(C|f) = P(g|f) P(W_1|...) P(A_1|...) P(X_1|...) ... P(L|...)$

where the crystal structure $C$ is represented by the sequence $g-(W_{i}-A_{i}-X_{i})_{n}-L$:
- $f$: chemical formula, e.g. Cu<sub>12</sub>Sb<sub>4</sub>S<sub>13</sub>
- $g$: space group number 1-230
- $W$: Wyckoff letter ('a', 'b', ...,'A')
- $A$: atom type ('H', 'He', ..., 'Og') in the chemical formula
- $X$: fractional coordinates
- $L$: lattice vector [a, b, c, alpha, beta, gamma]
- $P(W_i| ...)$ and $P(A_i| ...)$ are categorical distributions.
- $P(X_i| ...)$ is the mixture of von Mises distribution.
- $P(L| ...)$ is the mixture of Gaussian distribution.

We only consider symmetry inequivalent atoms in the crystal representation. The remaining atoms are restored based on the information of space group and Wyckoff letters. There is a natural alphabetical ordering for the Wyckoff letters, starting with 'a' for a position with the site-symmetry group of maximal order and ending with the highest letter for the general position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). Only for the cases where the Wyckoff letter cannot fully determine the structure, one needs to further consider fractional coordinates in the loss or sampling. 

## Status

Major milestones are summarized below.
- v0.6: _CrystalFormer_ for unified de novo generation and crystal structure prediction.
- v0.5: Initial release of _CrystalFormer-CSP_ for crystal structure prediction.
- v0.4.2 : Add implementation of direct preference optimization.
- v0.4.1 : Replace the absolute positional embedding with the Rotary Positional Embedding (RoPE).
- v0.4 : Add reinforcement learning (proximal policy optimization).
- v0.3 : Add conditional generation in the plug-and-play manner.
- v0.2 : Add Markov chain Monte Carlo (MCMC) sampling for template-based structure generation.
- v0.1 : Initial implementations of crystalline material generation conditioned on the space group.

## Get Started

**Notebooks**: The quickest way to get started with _CrystalFormer_ is our notebooks in the Google Colab platform:

- ColabCSP [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I7b5exbB2oBjexFIEaeDQexmYRDgLHVk?authuser=0#scrollTo=kfu6Ez9e6Sp7): Running _CrystalFormer-CSP_ Seamlessly on Google Colab
- CrystalFormer-RL [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ojSqMQzdnlWZRPOQP20nTvvIh67HXdwp#scrollTo=lKOZgUczOAxE) [![Open In Bohrium](https://cdn.dp.tech/bohrium/web/static/images/open-in-bohrium.svg)](https://bohrium.dp.tech/notebooks/52828216135): Reinforcement fine-tuning for materials design


<details>
<summary> Previous notebooks (only for reference but not actively maintained): </summary>

- CrystalFormer Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IMQV6OQgIGORE8FmSTmZuC5KgQwGCnDx?usp=sharing) [![Open In Bohrium](https://cdn.dp.tech/bohrium/web/static/images/open-in-bohrium.svg)](https://nb.bohrium.dp.tech/detail/68177247598): GUI notebook demonstrating the conditional generation of crystalline materials with _CrystalFormer_
- CrystalFormer Application [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QdkELaQXAHR1zEu2fcdfgabuoP61_wbU?usp=sharing): Generating stable crystals with a given structure prototype. This workflow can be applied to tasks that are dominated by element substitution

</details>

## Installation

Create a new environment and install the required packages, we recommend using python `3.10.*` and conda to create the environment:

```bash
  conda create -n crystalgpt python=3.10
  conda activate crystalgpt
```

Before installing the required packages, you need to install `jax` and `jaxlib` first.

### CPU installation

```bash
pip install -U "jax[cpu]"
```

### CUDA (GPU) installation

If you intend to use CUDA (GPU) to speed up the training, it is important to install the appropriate version of `jax` and `jaxlib`. It is recommended to check the [jax docs](https://github.com/google/jax?tab=readme-ov-file#installation) for the installation guide. The basic installation command is given below:

```bash
pip install --upgrade pip

# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install -U "jax[cuda12]"
```

### Install required packages and command line tools

After installing `jax` and `jaxlib`, you need to install the `crystalformer` package:

```bash
pip install .
```

During installation, the command line tools in the [cli](crystalformer/cli/) directory will be automatically installed.


## Available Weights

We release the weights of the model trained on the [Alex20s](https://huggingface.co/datasets/zdcao/alex-20s) dataset. More details are available in the [model card](./MODEL_CARD.md).

## Crystal Structure Prediction

<div align="center">
  <img align="middle" src="imgs/csp.png" width="500" alt="logo"/>
  <h2>Thinking fast and slow for crystal structure prediction</h2>
</div>


### Sample

```bash
python ./main.py --optimizer none --restore_path RESTORE_PATH --K 40 --num_samples 1000 --formula Cu12Sb4S13 --save_path SAVE_PATH
```

- `optimizer`: the optimizer to use, `none` means no training, only sampling
- `restore_path`: the path to the model weights
- `K`: the top-K number of space groups will be sampled uniformly. 
- `num_samples`: the number of samples to generate
- `formula`: the chemical formula 
- `save_path`: [Optional] the path to save the generated structures, if not provided, the structures will be saved in the `RESTORE_PATH` folder.

Instead of providing `K` for top-K sampling, you may directly provide your favorite space group number
- `spacegroup`: the space group number [1-230]

The sampled structure will be saved in the `SAVE_PATH/output_Cu12Sb4S13.csv` file. To transform the generated structure from `g, W, A, X, L` to the `cif` format, you can use the following command

```bash
python ./scripts/awl2struct.py --output_path SAVE_PATH --formula FORMULA 
```

- `output_path`: the path to read the generated `L, W, A, X` and save the `cif` files
- `formula`: the chemical formula constrained in the structure

This will save the generated structures in the `cif` format to a `output_Cu12Sb4S13_struct.csv` file. 

### Relax generated structures with MLFF:

```bash
python scripts/mlff_relax.py \
    --restore_path SAVE_PATH \
    --filename output_Cu12Sb4S13_struct.csv \
    --model orb-v3-conservative-inf-mpa \
    --model_path path/to/orb-v3.ckpt \
    --relaxation
```
This will produce relaxed structures in `relaxed_structures` with predicted energies.

### Energy Above Hull (Ehull)

Compute Ehull for all relaxed structures:

```bash
python scripts/e_above_hull_alex.py \
    --convex_path convex_hull_pbe.json.bz2 \
    --restore_path SAVE_PATH \
    --filename relaxed_structures.csv
```
### End-to-End Pipeline

Run sampling → CIF conversion → relaxation → Ehull ranking:

```bash
./postprocess.sh \
    -r RESTORE_PATH \
    -k 40 \
    --relaxation true \ 
    -n 1000 \
    -f Cu12Sb4S13 \ 
    -s SAVE_PATH
```

In case you are curious about the parameters, run:
```bash 
./postprocess.sh -h 
```

### Model Context Protocol (MCP) Server

_CrystalFormer_ can be easily integrated with AI assistants via the Model Context Protocol (MCP). Please refer to the [MCP README](./mcp/README.md) for detailed instructions on setting up and using the MCP server for crystal structure prediction.

## De Novo Generation

### Sample

```bash
python ./main.py --optimizer none --restore_path YOUR_MODEL_PATH --spacegroup 160 --num_samples 1000  --batchsize 1000 --temperature 1.0
```

- `optimizer`: the optimizer to use, `none` means no training, only sampling
- `restore_path`: the path to the model weights
- `spacegroup`: the space group number to sample
- `num_samples`: the number of samples to generate
- `batchsize`: the batch size for sampling
- `temperature`: the temperature for sampling

The sampling results will be saved in the `output_LABEL.csv` file, where the `LABEL` is the space group number `g` specified in the command `--spacegroup`.

### Evaluation

Before evaluating the generated structures, you need to transform the generated `g, W, A, X, L` to the `cif` format. You can use the following command to transform the generated structures to the `cif` format and save as the `csv` file:

```bash
python ./scripts/awl2struct.py --output_path YOUR_PATH --label SPACE_GROUP  --num_io_process 40
```

- `output_path`: the path to read the generated `L, W, A, X` and save the `cif` files
- `label`: the label to save the `cif` files, which is the space group number `g`
- `num_io_process`: the number of processes

> [!IMPORTANT]
> The following evaluation script requires the [`SMACT`](https://github.com/WMD-group/SMACT), [`matminer`](https://github.com/hackingmaterials/matminer), and [`matbench-genmetrics`](https://github.com/sparks-baird/matbench-genmetrics) packages. We recommend installing them in a separate environment to avoid conflicts with other packages.

Calculate the structure and composition validity of the generated structures:

```bash
python ./scripts/compute_metrics.py --root_path YOUR_PATH --filename YOUR_FILE --num_io_process 40
```

- `root_path`: the path to the dataset
- `filename`: the filename of the generated structures
- `num_io_process`: the number of processes

Calculate the novelty and uniqueness of the generated structures:

```bash
python ./scripts/compute_metrics_matbench.py --train_path TRAIN_PATH --test_path TEST_PATH --gen_path GEN_PATH --output_path OUTPUT_PATH --label SPACE_GROUP --num_io_process 40
```

- `train_path`: the path to the training dataset
- `test_path`: the path to the test dataset
- `gen_path`: the path to the generated dataset
- `output_path`: the path to save the metrics results
- `label`: the label to save the metrics results, which is the space group number `g`
- `num_io_process`: the number of processes

Note that the training, test, and generated datasets should contain the structures within the **same** space group `g` which is specified in the command `--label`.

More details about post-processing are available in the [scripts](./scripts/README.md) folder.

## Advanced usage

### Reinforcement Fine-tuning

> [!IMPORTANT]
> Before running the reinforcement fine-tuning, please make sure you have installed the corresponding machine learning force field model or property prediction model. The `mlff_model` and `mlff_path` arguments in the command line should be set according to the model you are using. Currently, we only support the [`orb`](https://github.com/orbital-materials/orb-models) model for the $E_{hull}$ reward. [`BatchRelaxer`](https://github.com/zdcao121/BatchRelaxer) is also needed for batch structure relaxation during fine-tuning.


```bash
train_ppo --folder ./data/\
          --restore_path YOUR_PATH\
          --reward ehull\
          --convex_path YOUR_PATH/convex_hull_pbe.json.bz2\
          --mlff_model orb-v3-conservative-inf-mpa\
          --mlff_path YOUR_PATH/orb-v3-conservative-inf-mpa-20250404.ckpt \
          --lr 1e-05 \
          --dropout_rate 0.0 \
          --K 40 \
          --batchsize 500 \
          --formula LiPH2O4 
```

where
- `folder`: the folder to save the model and logs
- `restore_path`: the path to the pre-trained model weights
- `reward`: the reward function to use, `ehull` means the energy above the convex hull
- `convex_path`: the path to the convex hull data, which is used to calculate the $E_{hull}$. Only used when the reward is `ehull`
- `mlff_model`: the machine learning force field model to predict the total energy. We support [`orb`](https://github.com/orbital-materials/orb-models) model for the $E_{hull}$ reward
- `mlff_path`: the path to load the checkpoint of the machine learning force field model

The legacy CSP path supports the `ehull` reward; the Issue #68 path adds the
CPU-only `xrd` reward. For DNG reinforcement fine-tuning, simply omit the
`--formula` argument.

### Powder XRD inverse recovery

The Issue #68 XRD path keeps the pretrained CrystalFormer as the prior and
uses a host-side, non-differentiable pymatgen simulator as the reward.  A
candidate is expanded from `(G, L, XYZ, A, W)`, simulated with
`XRDCalculator`, and scored by one-to-one peak matching. The scorer searches a
global q-space scale and a small zero shift, then combines matched, missing,
and extra peaks into a weighted F1 score. XRD similarity is a maximize reward;
the existing e-hull/property
rewards keep their original minimize convention.  The simulator never runs
inside a JAX gradient transformation.

The default peak settings were calibrated on 13 self-supervised crystal cases:
height/prominence `0.08/0.05`, smoothing width `0.10` degrees, intrinsic peak
width at least `0.05` degrees, minimum separation `0.15` degrees, at most `40`
peaks, q-position tolerance `0.04`, global scale range `0.70-1.40`, and zero
shift range `+-0.03` in q. These are deliberately modest defaults: weak noise is
ignored, while missing and extra peaks still lower the score.

Create a reproducible simulated target from a known CIF:

```bash
python issue68_xrd_recovery/scripts/xrd_recovery.py target \
  --cif known.cif \
  --output experiments/xrd/target.csv \
  --grid-step 0.05 \
  --fwhm 0.10 \
  --noise-std 0.01 \
  --seed 0
```

For a constrained CPU/WSL smoke run, keep the candidate batch and sampling
multiplier small and run one job at a time:

```bash
JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m crystalformer.cli.train_ppo \
  --reward xrd \
  --formula Si \
  --xrd_target experiments/xrd/target.csv \
  --restore_path BASE_CHECKPOINT \
  --folder experiments/xrd/ppo/ \
  --safe_cpu --epochs 1 --ppo_epochs 1 --batchsize 4 \
  --sample_multiplier 2 --max_sampling_attempts 20 --K 5 \
  --seed 42 --sg_temperature 0.8 --sg_epsilon 0.05 \
  --h0_size 64 --transformer_layers 2 --num_heads 4 \
  --key_size 16 --model_size 64 --embed_size 64
```

The checkpoint must have the same compact architecture. `--safe_cpu` refuses
the default 16-layer/256-dimensional configuration and checkpoints larger than
64 MiB before JAX allocation instead of risking a memory spike; use the full
pretrained checkpoint on a machine with more RAM.

Score candidate CIFs without loading the Transformer or an MLFF:

```bash
python issue68_xrd_recovery/scripts/xrd_recovery.py evaluate \
  --target experiments/xrd/target.csv \
  --candidates experiments/xrd/candidates/ \
  --ground-truth known.cif \
  --output experiments/xrd/results.csv
```

The evaluation reports peak-matching similarity and `StructureMatcher` recovery
separately; a high powder-pattern score alone is not a structure-identity
claim.

Run the no-prior random-move Monte Carlo reference with a bounded budget:

```bash
python issue68_xrd_recovery/scripts/xrd_recovery.py random-move \
  --target experiments/xrd/target.csv \
  --formula Si \
  --output-dir experiments/xrd/random_move/ \
  --ground-truth known.cif \
  --evaluations 500 --restarts 4 --seed 0
```

Generate a reproducible tau sweep command file and summarize completed runs:

```bash
python issue68_xrd_recovery/scripts/xrd_study.py make-commands \
  --target experiments/xrd/target.csv --formula Si \
  --restore-path BASE_CHECKPOINT --output-root experiments/xrd/tau_sweep \
  --seeds 42,43 --ground-truth known.cif
python issue68_xrd_recovery/scripts/xrd_study.py summarize \
  --runs experiments/xrd --output experiments/xrd/study
```

`--tau` is an alias for the existing PPO `--beta` KL/prior coefficient. The
study tool reports similarity-threshold recovery separately from
`StructureMatcher`; with `--ground-truth`, each generated serial command also
evaluates its candidate CIFs and records the matching metadata. It does not
treat a good spectrum fit as proof of a ground-truth structure. `--seed` makes
repeated runs reproducible; `--sg_temperature` and `--sg_epsilon` control
space-group sampling, `--exploration_weight` aliases the entropy weight, and
`--diversity_weight` enables an optional inverse-frequency bonus for distinct
discrete structure sequences.

On a small WSL machine, set `JAX_PLATFORMS=cpu`, skip the unavailable CUDA
probe with `JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1`, disable XLA preallocation, use
a batch size of at most 8, and run one sweep value at a time. The
random-move and evaluation commands do not load a checkpoint and are the safe
way to validate the forward model before attempting PPO.

### Writing custom reward functions

Custom reward functions are implemented as Python factory functions that return a pair (`reward_fn`, `batch_reward_fn`). Follow the patterns in [crystalformer/reinforce/reward.py](crystalformer/reinforce/reward.py) to implement your own reward functions.

> [!CAUTION]
> **Reward direction**: Legacy energy/property rewards use the historical
> minimize convention. The XRD path explicitly sets `reward_direction="maximize"`
> so a larger peak-matching similarity is better; do not negate the XRD score.

Guidelines

- Signature: `reward_fn(x)` accepts a single sample tuple `(G, L, XYZ, A, W)` and returns a scalar reward (float or numpy scalar).
- Batch API: `batch_reward_fn(x)` accepts a batched `x=(G,L,XYZ,A,W)` (JAX arrays). It should convert inputs to CPU numpy arrays, compute per-sample rewards (e.g., by calling reward_fn or a vectorized routine), and return a jax.numpy array placed on the GPU (see examples below for device transfers using jax.device_put). The XRD factory is the deliberate host-side exception and returns a NumPy vector so the simulator stays outside JAX transformations.
- Structure conversion: use `get_atoms_from_GLXYZAW(G, L, XYZ, A, W)` from crystalformer.reinforce.reward to convert the representation to ASE Atoms or a pymatgen Structure before calling property predictors or MLFFs.
- Robustness: catch exceptions and return a sensible dummy or clipped reward for failed predictions to avoid crashing training.
- Performance: for heavy operations (relaxations, MLFF evaluations), prefer parallel/batched implementations where possible.

Minimal example

```python
from crystalformer.reinforce import reward as reward_mod
from pymatgen.io.ase import AseAtomsAdaptor
import jax
import jax.numpy as jnp
import numpy as np

def make_custom_reward_fn(model, dummy=0.0):
    ase_adaptor = AseAtomsAdaptor()

    def reward_fn(x):
        G, L, XYZ, A, W = x
        try:
            atoms = reward_mod.get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            struct = ase_adaptor.get_structure(atoms)
            val = model(struct)  # compute property from structure
            return float(val)
        except Exception:
            return float(dummy)

    def batch_reward_fn(x):
        # move data to CPU numpy for Python-side processing
        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        # iterate over samples (or use parallel map) and collect rewards
        output = [reward_fn(sample) for sample in zip(*x)]
        # return jnp.array on GPU
        return jax.device_put(jnp.array(output), jax.devices('gpu')[0]).block_until_ready()

    return reward_fn, batch_reward_fn
```

See the concrete implementations in [crystalformer/reinforce/reward.py](crystalformer/reinforce/reward.py) for more complete patterns (device transfers, relaxation, parallelization and clipping of rewards).

### Pretrain

```bash
python ./main.py --folder ./data/ --cfg_drop_prob 0.5 --train_path YOUR_PATH/alex20s/train.csv --valid_path YOUR_PATH/alex20s/val.csv
```
where
- `folder`: the folder to save the model and logs
- `cfg_drop_prob`: classifier-free guidance drop probability for formula conditioning. A value of `1` disables formula conditioning (DNG), while a value of `0` always enables formula conditioning (CSP)
- `train_path`: the path to the training dataset
- `valid_path`: the path to the validation dataset


Test the prediction accuracy of space groups on the test dataset

```bash 

python scripts/predict_g.py --restore_path YOUR_PATH --valid_path YOUR_PATH --Nf 5 --Kx 16 --Kl 4 --h0_size 256 --transformer_layers 16 --num_heads 8 --key_size 32 --model_size 256 --embed_size 256 --batchsize 1000
```


## How to cite

```bibtex
@article{cao2024space,
      title={Space Group Informed Transformer for Crystalline Materials Generation}, 
      author={Zhendong Cao and Xiaoshan Luo and Jian Lv and Lei Wang},
      year={2024},
      eprint={2403.15734},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

```bibtex
@article{cao2025crystalformerrl,
      title={CrystalFormer-RL: Reinforcement Fine-Tuning for Materials Design}, 
      author={Zhendong Cao and Lei Wang},
      year={2025},
      eprint={2504.02367},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2504.02367}, 
}
```

```bibtex
@misc{cao2025crystalformercsp,
      title={CrystalFormer-CSP: Thinking Fast and Slow for Crystal Structure Prediction}, 
      author={Zhendong Cao and Shigang Ou and Lei Wang},
      year={2025},
      eprint={2512.18251},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2512.18251}, 
}
```

**Note**: This project is unrelated to https://github.com/omron-sinicx/crystalformer with the same name.
