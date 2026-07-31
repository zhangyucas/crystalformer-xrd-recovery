import os
import json
import gc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def _resolve_xrd_args(args, parser):
    """Validate XRD inputs and resolve CLI values against target metadata."""

    if args.formula is None:
        parser.error('--formula is required for the conditional xrd reward')
    if args.xrd_target is None and args.xrd_target_structure is None:
        parser.error('--xrd_target or --xrd_target_structure is required for --reward xrd')
    if args.xrd_target is not None and args.xrd_target_structure is not None:
        parser.error('pass only one of --xrd_target and --xrd_target_structure')
    for option_name, option_value in (
        ('--xrd_target', args.xrd_target),
        ('--xrd_target_structure', args.xrd_target_structure),
    ):
        if option_value is not None and not os.path.isfile(option_value):
            parser.error(f'{option_name} does not exist: {option_value}')

    metadata = {}
    if args.xrd_target is not None:
        metadata_path = Path(f'{args.xrd_target}.json')
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f'could not read XRD metadata {metadata_path}: {exc}')

    def xrd_value(argument_name, metadata_name, fallback):
        value = getattr(args, argument_name)
        return metadata.get(metadata_name, fallback) if value is None else value

    args.xrd_wavelength = xrd_value('xrd_wavelength', 'wavelength', 'CuKa')
    args.xrd_two_theta_min = xrd_value('xrd_two_theta_min', 'two_theta_min', 5.0)
    args.xrd_two_theta_max = xrd_value('xrd_two_theta_max', 'two_theta_max', 90.0)
    args.xrd_grid_step = xrd_value('xrd_grid_step', 'grid_step', 0.05)
    args.xrd_profile = xrd_value('xrd_profile', 'profile', 'gaussian')
    args.xrd_fwhm = xrd_value('xrd_fwhm', 'fwhm', 0.10)
    args.xrd_eta = xrd_value('xrd_eta', 'eta', 0.5)
    args.xrd_target_is_peaks = xrd_value('xrd_target_is_peaks', 'target_is_peaks', False)
    if isinstance(args.xrd_target_is_peaks, str):
        lowered = args.xrd_target_is_peaks.strip().lower()
        if lowered in {'true', '1', 'yes'}:
            args.xrd_target_is_peaks = True
        elif lowered in {'false', '0', 'no'}:
            args.xrd_target_is_peaks = False
        else:
            parser.error('--xrd_target_is_peaks metadata must be boolean')


def _validate_safe_checkpoint(args, parser):
    """Resolve and reject a full-size checkpoint before JAX allocation.

    ``checkpoint.find_ckpt_filename`` chooses directory entries by reverse
    filename order and briefly unpickles the selected file.  Resolve the same
    entry here so the size guard cannot inspect one checkpoint and later load
    another, and so the loader does not need a second directory probe.
    """

    if not getattr(args, "safe_cpu", False) or args.restore_path is None:
        return
    restore_path = Path(args.restore_path)
    if restore_path.is_file():
        candidates = [restore_path]
    elif restore_path.is_dir():
        candidates = sorted(
            restore_path.glob("epoch_*.pkl"),
            key=lambda item: item.name,
            reverse=True,
        )
    else:
        return
    if not candidates:
        return
    checkpoint_path = candidates[0]
    checkpoint_size = checkpoint_path.stat().st_size
    max_bytes = 64 * 1024 * 1024
    if checkpoint_size > max_bytes:
        parser.error(
            f'safe_cpu refuses checkpoint {checkpoint_path} '
            f'({checkpoint_size / (1024 ** 2):.1f} MiB > {max_bytes / (1024 ** 2):.0f} MiB); '
            'use a compact architecture-matched checkpoint on this host'
        )
    # Passing the explicit file avoids the directory scanner unpickling the
    # checkpoint once for validation and then loading it again immediately.
    args.restore_path = str(checkpoint_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('training parameters')
    group.add_argument('--epochs', type=int, default=100, help='')
    group.add_argument('--batchsize', type=int, default=100, help='')
    group.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    group.add_argument('--lr_decay', type=float, default=0.0, help='lr decay')
    group.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    group.add_argument('--clip_grad', type=float, default=1.0, help='clip gradient')
    group.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="optimizer type")

    group.add_argument("--folder", default="./data/", help="the folder to save data")
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")

    group.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    group.add_argument('--sample_multiplier', type=float, default=10.0,
                       help='conditional sampling multiplier; lower it for low-memory runs')
    group.add_argument('--sampling_batchsize', '--sampling-batchsize', type=int, default=None,
                       help='optional fixed sampler batch, independent of the PPO batch')
    group.add_argument('--ppo_microbatch_size', '--ppo-microbatch-size', type=int, default=None,
                       help='accumulate PPO gradients in fixed-size microbatches')
    group.add_argument('--max_sampling_attempts', type=int, default=1000,
                       help='maximum conditional sampling attempts per epoch')
    group.add_argument('--composition_max_atoms', '--composition-max-atoms', type=int, default=512,
                       help='maximum unit-cell atoms considered by composition reachability')
    group.add_argument('--composition_size_bias', '--composition-size-bias', type=float, default=0.0,
                       help='soft preference for smaller reachable unit cells; 0 disables it')
    group.add_argument('--seed', type=int, default=42,
                       help='random seed for model initialization and PPO sampling')
    group.add_argument('--safe_cpu', action='store_true',
                       help='force CPU, reject large models, and cap epochs/batch/sampling attempts')
    group.add_argument('--enable_x64', '--enable-x64', action='store_true',
                       help='use float64 JAX intermediates; disabled by default to reduce memory')
    group.add_argument('--reset_optimizer', '--reset-optimizer', action='store_true',
                       help='discard a checkpoint optimizer state and initialize a new one')

    group = parser.add_argument_group('transformer parameters')
    group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
    group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
    group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
    group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the first atom, 0 means we simply use a table for first aw_logit')
    group.add_argument('--transformer_layers', type=int, default=16, help='The number of layers in transformer')
    group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
    group.add_argument('--key_size', type=int, default=32, help='The key size')
    group.add_argument('--model_size', type=int, default=256, help='The model size')
    group.add_argument('--embed_size', type=int, default=256, help='The enbedding size')
    group.add_argument('--dropout_rate', type=float, default=0.1, help='The dropout rate for MLP')
    group.add_argument('--attn_dropout', type=float, default=0.1, help='The dropout rate for attention')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    group = parser.add_argument_group('sampling parameters')
    group.add_argument('--formula', type=str, default=None, help='chemical formula of the compound')
    group.add_argument('--K', type=int, default=0, help='top K number of space groups. 0 means we sample spacegroup')
    group.add_argument('--spacegroup', type=int, default=None, help='the spacegroup number 1-230')
    group.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
    group.add_argument('--temperature', type=float, default=1.0,
                       help='temperature used for atom, coordinate, and lattice sampling')
    group.add_argument('--sg_temperature', '--sg-temperature', type=float, default=None,
                       help='optional temperature used only for space-group sampling')
    group.add_argument('--sg_epsilon', '--sg-epsilon', type=float, default=0.0,
                       help='epsilon-greedy exploration probability for space-group sampling')

    group = parser.add_argument_group('reinforcement learning parameters')
    group.add_argument('--reward', type=str, default='ehull', choices=['ehull', 'prop', 'dielectric', 'density', 'xrd'], help='reward function to use')
    group.add_argument('--relaxation', action='store_true', help='whether to relax the structures')
    group.add_argument('--convex_path', type=str, default='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/convex_hull_pbe.json.bz2')
    group.add_argument('--alpha', '--exploration_weight', '--exploration-weight', dest='alpha', type=float, default=0.1,
                       help='entropy/exploration weight (exploration_weight is the Issue #68 alias)')
    group.add_argument('--beta', '--tau', dest='beta', type=float, default=0.1,
                       help='KL/prior strength tau (beta is the historical option name)')
    group.add_argument('--gamma', type=float, default=1.0, help='weight for experience buffer')
    group.add_argument('--diversity_weight', '--diversity-weight', type=float, default=0.0,
                       help='optional inverse-frequency bonus for distinct sampled (G,W,A) sequences')
    group.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    group.add_argument('--ehull_clip', type=float, default=20, help='clip parameter for ehull value')
    group.add_argument('--ppo_epochs', type=int, default=5, help='number of PPO epochs')
    group.add_argument('--mlff_model', type=str, default='orb-v3-conservative-inf-mpa', choices=['orb-v2', 'orb-v3-conservative-inf-mpa', 'orb-v3-direct-20-mpa', 'matgl'], help='the model to use for RL reward')
    group.add_argument('--mlff_path', type=str, default='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/orb-v3-conservative-inf-mpa-20250404.ckpt', help='path to the MLFF model')

    group = parser.add_argument_group('powder XRD reward parameters')
    group.add_argument('--xrd_target', '--xrd_target_path', dest='xrd_target', default=None,
                       help='CSV pattern (two_theta,intensity) or a target CIF')
    group.add_argument('--xrd_target_structure', default=None,
                       help='optional target CIF; alternative to --xrd_target')
    group.add_argument('--xrd_wavelength', default=None,
                       help='pymatgen XRD wavelength, e.g. CuKa or a numeric Angstrom value')
    group.add_argument('--xrd_two_theta_min', type=float, default=None)
    group.add_argument('--xrd_two_theta_max', type=float, default=None)
    group.add_argument('--xrd_grid_step', type=float, default=None)
    group.add_argument('--xrd_profile', choices=['gaussian', 'pseudo-voigt'], default=None)
    group.add_argument('--xrd_fwhm', type=float, default=None,
                       help='profile full width at half maximum in degrees')
    group.add_argument('--xrd_eta', type=float, default=None,
                       help='Lorentzian fraction for pseudo-Voigt profiles')
    group.add_argument('--xrd_target_is_peaks', action='store_true', default=None,
                       help='treat target CSV rows as discrete peaks before broadening')

    group = parser.add_argument_group('loss parameters')
    group.add_argument("--lamb_a", type=float, default=1.0, help="weight for the a part")
    group.add_argument("--lamb_g", type=float, default=1.0, help="weight for the g partc")
    group.add_argument("--lamb_xyz", type=float, default=1.0, help="weight for the xyz part")
    group.add_argument("--lamb_w", type=float, default=1.0, help="weight for the w part")
    group.add_argument("--lamb_l", type=float, default=1.0, help="weight for the lattice part")

    group = parser.add_argument_group('property reward parameters')
    group.add_argument('--target', type=float, default=-3, help='target property value to optimize')
    group.add_argument('--dummy_value', type=float, default=0, help='dummy value for the property')
    group.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mae'], help='loss type for the property reward')

    args = parser.parse_args()

    if args.epochs <= 0 or args.ppo_epochs <= 0 or args.batchsize <= 0:
        parser.error('--epochs, --ppo_epochs and --batchsize must be positive')
    if args.sample_multiplier <= 0 or args.max_sampling_attempts <= 0:
        parser.error('--sample_multiplier and --max_sampling_attempts must be positive')
    if args.composition_max_atoms <= 0:
        parser.error('--composition-max-atoms must be positive')
    if args.composition_size_bias < 0:
        parser.error('--composition-size-bias cannot be negative')
    if args.sampling_batchsize is not None and args.sampling_batchsize <= 0:
        parser.error('--sampling_batchsize must be positive')
    if args.ppo_microbatch_size is not None and args.ppo_microbatch_size <= 0:
        parser.error('--ppo_microbatch_size must be positive')
    if args.ppo_microbatch_size is not None and args.batchsize % args.ppo_microbatch_size:
        parser.error('--batchsize must be divisible by --ppo_microbatch_size')
    if not 0.0 < args.top_p <= 1.0 or args.temperature <= 0:
        parser.error('--top_p must be in (0, 1] and --temperature must be positive')
    if args.sg_temperature is not None and args.sg_temperature <= 0:
        parser.error('--sg_temperature must be positive')
    if not 0.0 <= args.sg_epsilon <= 1.0:
        parser.error('--sg_epsilon must be in [0, 1]')
    if args.beta < 0:
        parser.error('--beta/--tau cannot be negative')
    if args.alpha < 0 or args.gamma < 0 or args.diversity_weight < 0:
        parser.error('--alpha, --gamma and --diversity_weight cannot be negative')

    if args.reward == 'xrd':
        _resolve_xrd_args(args, parser)

    if args.restore_path is not None and not os.path.exists(args.restore_path):
        parser.error(f'--restore_path does not exist: {args.restore_path}')

    if args.safe_cpu:
        safe_limits = {
            'transformer_layers': 4,
            'num_heads': 8,
            'key_size': 32,
            'model_size': 128,
            'embed_size': 128,
            'h0_size': 128,
        }
        oversized = [
            f'{name}={getattr(args, name)} (limit {limit})'
            for name, limit in safe_limits.items()
            if getattr(args, name) > limit
        ]
        if oversized:
            parser.error(
                'safe_cpu refuses a large Transformer configuration: '
                + ', '.join(oversized)
                + '; use a compact checkpoint/configuration for this host'
            )
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.25'
        os.environ['JAX_PLATFORMS'] = 'cpu'
        # Some JAX CUDA plugin builds probe ``cuInit`` even when CPU is the
        # only requested backend.  Skip that optional probe so a CPU run does
        # not emit a misleading CUDA traceback or touch the unavailable GPU.
        os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK', '1')
        # Keep BLAS/JAX host threads bounded as well as device memory.  Use
        # ``setdefault`` so a caller can deliberately choose another limit.
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ.setdefault(variable, '1')
        args.epochs = min(args.epochs, 2)
        args.ppo_epochs = min(args.ppo_epochs, 1)
        args.batchsize = min(args.batchsize, 8)
        args.sample_multiplier = min(args.sample_multiplier, 2.0)
        args.max_sampling_attempts = min(args.max_sampling_attempts, 50)
        print('safe_cpu enabled: bounded CPU-only PPO configuration')

    _validate_safe_checkpoint(args, parser)

    # Delay JAX and model imports until after argument validation and the safe
    # environment settings.  `--help` and bad paths therefore do not probe a
    # CUDA plugin or allocate model state.
    import jax
    import jax.numpy as jnp
    if args.safe_cpu:
        # Set the backend before any model/module code can query devices.
        try:
            jax.config.update('jax_platforms', 'cpu')
        except Exception as exc:
            print(f'warning: JAX backend was already initialized ({exc}); continuing with the existing backend')
    jax.config.update("jax_enable_x64", args.enable_x64)
    from functools import partial
    import optax
    from crystalformer.src.loss import make_loss_fn
    from crystalformer.src.transformer import make_transformer
    from crystalformer.src.sample import make_sample_crystal
    from crystalformer.src.formula import formula_string_to_composition_vector
    import crystalformer.src.checkpoint as checkpoint
    from crystalformer.reinforce.ppo import train, make_ppo_loss_fn

    
    print("================ parameters ================")
    # print all the parameters
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


    print("\n========== Prepare transformer ==========")
    ################### Model #############################
    key = jax.random.PRNGKey(args.seed)
    ckpt = None
    ckpt_filename = None
    epoch_finished = 0
    if args.restore_path is not None:
        ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path)
        if ckpt_filename is not None:
            print("Load checkpoint file early: %s, epoch finished: %g" % (ckpt_filename, epoch_finished))
            ckpt = checkpoint.load_data(ckpt_filename)
            params = ckpt["params"]
            _, transformer = make_transformer(
                key, args.Nf, args.Kx, args.Kl, args.n_max,
                args.h0_size,
                args.transformer_layers, args.num_heads,
                args.key_size, args.model_size, args.embed_size,
                args.atom_types, args.wyck_types,
                args.dropout_rate, args.attn_dropout,
                initialize=False,
            )
        else:
            params, transformer = make_transformer(
                key, args.Nf, args.Kx, args.Kl, args.n_max,
                args.h0_size,
                args.transformer_layers, args.num_heads,
                args.key_size, args.model_size, args.embed_size,
                args.atom_types, args.wyck_types,
                args.dropout_rate, args.attn_dropout,
            )
    else:
        params, transformer = make_transformer(
            key, args.Nf, args.Kx, args.Kl, args.n_max,
            args.h0_size,
            args.transformer_layers, args.num_heads,
            args.key_size, args.model_size, args.embed_size,
            args.atom_types, args.wyck_types,
            args.dropout_rate, args.attn_dropout,
        )

    transformer_name = 'Nf_%d_Kx_%d_Kl_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_e_%d_drop_%g'%(args.Nf, args.Kx, args.Kl, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size, args.embed_size, args.dropout_rate)

    parameter_count = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))
    print ("# of transformer params", parameter_count)

    if args.formula is not None:
        composition = formula_string_to_composition_vector(args.formula)
    else:
        composition = jnp.zeros((args.atom_types,), dtype=int)
    print ('composition vector of', args.formula)
    print (composition)

    ################### Train #############################

    loss_fn, logp_fn = make_loss_fn(
        args.n_max,
        args.atom_types,
        args.wyck_types,
        args.Kx,
        args.Kl,
        transformer,
        composition_reachability=args.formula is not None,
        composition_max_atoms=args.composition_max_atoms,
        composition_size_bias=args.composition_size_bias,
    )

    print("\n========== Prepare logs ==========")

    reward_tag = 'xrd' if args.reward == 'xrd' else args.mlff_model
    effective_sg_temperature = args.temperature if args.sg_temperature is None else args.sg_temperature
    if args.optimizer != "none" or args.restore_path is None:
        output_name = "%s_%s_ppo_%d_a_%g_b_%g_c_%g_T_%g_" % (args.formula, reward_tag, args.ppo_epochs, args.alpha, args.beta, args.gamma, args.temperature) \
                    + ("spg_%d_" %args.spacegroup if args.spacegroup is not None else "K_%d_"%args.K) \
                    + ("relax_" if args.relaxation else "") \
                    + ("eclip_%g_"%(args.ehull_clip) ) \
                    + ('g_%g_w_%g_a_%g_xyz_%g_l_%g_'%(args.lamb_g, args.lamb_w, args.lamb_a, args.lamb_xyz, args.lamb_l)) \
                    + args.optimizer+"_bs_%d_lr_%g" % (args.batchsize, args.lr) \
                    + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                    + ("_sizebias_%g" % args.composition_size_bias if args.composition_size_bias else "") \
                    +  "_sgT_%g_sgE_%g_div_%g_seed_%d_" % (
                        effective_sg_temperature,
                        args.sg_epsilon,
                        args.diversity_weight,
                        args.seed,
                    ) + transformer_name
        output_path = os.path.join(args.folder, output_name)

        os.makedirs(output_path, exist_ok=True)
        print("Create directory for output: %s" % output_path)
    else:
        output_path = os.path.dirname(args.restore_path)
        print("Will output samples to: %s" % output_path)

    # Keep every sweep self-describing without loading the large checkpoint
    # again during post-processing.  The file is intentionally small.
    config_path = os.path.join(output_path, "run_config.json")
    config_values = vars(args).copy()
    # Preserve the Issue #68 vocabulary alongside the historical argparse
    # destinations so downstream sweep tools need not infer aliases.
    config_values["tau"] = args.beta
    config_values["exploration_weight"] = args.alpha
    with open(config_path, "w") as config_file:
        json.dump(config_values, config_file, indent=2, sort_keys=True)


    print("\n========== Load checkpoint==========")
    if ckpt is None:
        ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or output_path)
    if ckpt is None and ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    elif ckpt is None:
        print("No checkpoint file found. Start from scratch.")

    schedule = lambda t: args.lr/(1+args.lr_decay*t)

    if args.optimizer == "adam":
        optimizer = optax.chain(optax.clip_by_global_norm(args.clip_grad), 
                                optax.scale_by_adam(), 
                                optax.scale_by_schedule(schedule), 
                                optax.scale(-1.))
    elif args.optimizer == 'adamw':
        optimizer = optax.chain(optax.clip(args.clip_grad),
                                optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
                            )

    if ckpt is not None and "opt_state" in ckpt and not args.reset_optimizer:
        # The historical path initialized another pair of Adam moment trees and
        # immediately replaced them with this saved state.  Reusing the state
        # directly avoids that full-model transient allocation.
        opt_state = ckpt["opt_state"]
        print("Loaded optimizer state from checkpoint without duplicate initialization")
    else:
        if ckpt is not None:
            # Release the saved supervised-training moments before allocating
            # fresh PPO moments, rather than briefly retaining both full trees.
            ckpt.pop("opt_state", None)
            gc.collect()
        opt_state = optimizer.init(params)
        if args.reset_optimizer:
            print("Initialized a fresh optimizer state")
    del ckpt
    gc.collect()

    calc = None
    model = None
    if args.reward == "ehull":
        print("\n========== Load mlff ==========")
        print(f"Using {args.mlff_model} model at {args.mlff_path}")
        # Keep the ORB stack out of XRD and density-only runs.  Importing it
        # can initialize CUDA and consumes a substantial amount of memory.
        from orb_models.forcefield import pretrained
        if args.mlff_model in pretrained.ORB_PRETRAINED_MODELS:
            from orb_models.forcefield.calculator import ORBCalculator
            orbff = pretrained.ORB_PRETRAINED_MODELS[args.mlff_model](args.mlff_path, device='cuda')
            calc = ORBCalculator(orbff, device='cuda')
        else:
            raise NotImplementedError("the ehull path currently requires an ORB model")
    elif args.reward in {"prop", "dielectric"}:
        print("\n========== Load property model ==========")
        import matgl
        if ',' in args.mlff_path:
            import torch
            torch.set_default_dtype(torch.float32)
            model1 = matgl.load_model(args.mlff_path.split(',')[0]).predict_structure
            model2 = matgl.load_model(args.mlff_path.split(',')[1])
            model2 = partial(model2.predict_structure, state_attr=torch.tensor([0]))
            model = [model1, model2]
        else:
            model = matgl.load_model(args.mlff_path).predict_structure

    print("\n========== Load rl reward ==========")

    reward_direction = "minimize"
    metric_name = "e"
    if args.reward == "ehull":
        import bz2
        from crystalformer.reinforce.reward import make_ehull_reward_fn
        with bz2.open(args.convex_path) as fh:
            ref_data = json.loads(fh.read().decode('utf-8'))
            # remove 'structure' key in the 'entries' dictionary to reduce the size of the ref_data
            for entry in ref_data['entries']:
                entry.pop('structure')
                
        _, batch_reward_fn = make_ehull_reward_fn(calc, ref_data, n_jobs=args.num_io_process, 
                                                  relaxation=args.relaxation, clip_value=args.ehull_clip)
    
    elif args.reward == "prop":
        from crystalformer.reinforce.reward import make_prop_reward_fn
        _, batch_reward_fn = make_prop_reward_fn(model, args.target, args.dummy_value, args.loss_type)

    elif args.reward == "dielectric":
        assert len(model) == 2, "Two models are required for dielectric reward"
        from crystalformer.reinforce.reward import make_dielectric_reward_fn
        _, batch_reward_fn = make_dielectric_reward_fn(model, args.dummy_value)

    elif args.reward == "density":
        from crystalformer.reinforce.reward import make_density_reward_fn
        _, batch_reward_fn = make_density_reward_fn(inverse=False)

    elif args.reward == "xrd":
        from crystalformer.reinforce.reward import make_xrd_reward_fn
        _, batch_reward_fn = make_xrd_reward_fn(
            target=args.xrd_target,
            target_structure=args.xrd_target_structure,
            wavelength=args.xrd_wavelength,
            two_theta_range=(args.xrd_two_theta_min, args.xrd_two_theta_max),
            grid_step=args.xrd_grid_step,
            profile=args.xrd_profile,
            fwhm=args.xrd_fwhm,
            eta=args.xrd_eta,
            target_is_peaks=args.xrd_target_is_peaks,
        )
        reward_direction = "maximize"
        metric_name = "xrd"

    else:
        raise NotImplementedError

    print("\n========== Load sample function ==========")

    sample_crystal = make_sample_crystal(
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
        composition_reachability=args.formula is not None,
        composition_max_atoms=args.composition_max_atoms,
        composition_size_bias=args.composition_size_bias,
    )

    print("\n========== Start RL training ==========")
    ppo_loss_fn = make_ppo_loss_fn(logp_fn, args.eps_clip, beta=args.beta, alpha=args.alpha, gamma=args.gamma,
                                   lamb_g = args.lamb_g, lamb_a = args.lamb_a, lamb_w = args.lamb_w, lamb_xyz = args.lamb_xyz, lamb_l = args.lamb_l
                                   )

    # PPO training
    params, opt_state = train(
        key,
        optimizer,
        opt_state,
        logp_fn,
        batch_reward_fn,
        ppo_loss_fn,
        sample_crystal,
        composition,
        params,
        epoch_finished,
        args.epochs,
        args.ppo_epochs,
        args.batchsize,
        output_path,
        reward_direction=reward_direction,
        sample_multiplier=args.sample_multiplier,
        sampling_batchsize=args.sampling_batchsize,
        ppo_microbatch_size=args.ppo_microbatch_size,
        max_sampling_attempts=args.max_sampling_attempts,
        metric_name=metric_name,
        diversity_weight=args.diversity_weight,
    )


if __name__ == "__main__":
    main()
