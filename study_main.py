import jax
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os
import multiprocessing
import math
import pandas as pd
import numpy as np 
np.set_printoptions(threshold=np.inf)

from crystalformer.src.utils import GLXYZAW_from_file, letter_to_number
from crystalformer.src.elements import element_list
from crystalformer.src.transformer import make_transformer  
from crystalformer.src.train import train
from crystalformer.src.sample import make_sample_crystal
from crystalformer.src.loss import make_loss_fn
import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.formula import formula_string_to_composition_vector, find_composition_vector

import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def sampling_output_path(save_path, restore_path):
    if save_path is not None:
        return save_path
    if restore_path is None:
        return os.path.join(PROJECT_ROOT, "data", "samples")
    if os.path.isfile(restore_path):
        return os.path.dirname(restore_path)
    return restore_path


def configure_jax_platform(platform):
    if platform == "auto":
        print("JAX backend:", jax.default_backend())
        print("JAX devices:", jax.devices())
        return

    jax.config.update("jax_platform_name", platform)
    try:
        devices = jax.devices(platform)
    except RuntimeError as error:
        raise RuntimeError(
            f"Requested --platform {platform}, but JAX cannot initialize that backend. "
            "For NVIDIA GPUs, install CUDA-enabled JAX with `pip install -U \"jax[cuda12]\"` "
            "and make sure the GPU is visible to the operating system."
        ) from error

    if not devices:
        raise RuntimeError(f"Requested --platform {platform}, but JAX found no {platform} devices.")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {devices}")


parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('training parameters')
group.add_argument('--epochs', type=int, default=10000, help='') #我要梯度下降多少轮
group.add_argument('--batchsize', type=int, default=100, help='') #每个batch包含多少个样本
group.add_argument('--lr', type=float, default=1e-4, help='learning rate') #学习率
group.add_argument('--lr_decay', type=float, default=0.0, help='lr decay') #学习率衰减 因为我的越接近真实值梯度变化应越小，为了防止震荡，我们就让梯度的幅度慢慢变小
group.add_argument('--weight_decay', type=float, default=0.0, help='weight decay') #权重衰减，adamw的时候用到
group.add_argument('--clip_grad', type=float, default=1.0, help='clip gradient') #梯度裁剪	adam：全局梯度修剪：限制模长 adamw：元素梯度修剪，限制每个分量
group.add_argument("--optimizer", type=str, default="adam", choices=["none", "adam", "adamw"], help="optimizer type") #adam和adamw，adam是运用到了梯度的关系（保留一部分之前的一阶矩和二阶矩）为了防止梯度变化的太快，adamw再次之上还有一点点小优化
group.add_argument("--val_interval", type=int, default=100, help="validation interval") #每个比如100个epoch做一下记录作为日志
group.add_argument("--cfg_drop_prob", type=float, default=0.5, help="classifer-free guidance drop probability")

"""
cfg_drop_prob:有一定的概率把输入的化学式给mask掉,
	一部分样本：
	input = 化学式 composition + 晶体结构前文
	target = 晶体结构中的下一个变量
	
	另一部分样本：
	input = 空 composition + 晶体结构前文
	target = 晶体结构中的下一个变量

"""
group.add_argument("--folder", default=project_path("data"), help="the folder to save data")
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group('runtime parameters')
group.add_argument("--platform", default="auto", choices=["auto", "cpu", "gpu"], help="JAX platform to use")

group = parser.add_argument_group('dataset')
group.add_argument('--train_path', default=project_path("data", "mini.csv"), help='') #train用来训
group.add_argument('--valid_path', default=project_path("data", "mini.csv"), help='')  #训练过程中用来检查模型效果
group.add_argument('--test_path', default=project_path("data", "mini.csv"), help='') #训练完成之后用来最终评估模型的表现

group = parser.add_argument_group('transformer parameters')
group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the g and w of first atom') #g是空间群 w是wyckoff位置
group.add_argument('--transformer_layers', type=int, default=16, help='The number of layers in transformer')
group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
group.add_argument('--key_size', type=int, default=32, help='The key size') #query和key的维度
group.add_argument('--model_size', type=int, default=256, help='The model size')
group.add_argument('--embed_size', type=int, default=256, help='The enbedding size') #embedding vector的维度
group.add_argument('--dropout_rate', type=float, default=0.1, help='The dropout rate for MLP')
group.add_argument('--attn_dropout', type=float, default=0.1, help='The dropout rate for attention')

group = parser.add_argument_group('loss parameters')
group.add_argument("--lamb_a", type=float, default=1.0, help="weight for the a part relative to fc")
group.add_argument("--lamb_w", type=float, default=1.0, help="weight for the w part relative to fc")
group.add_argument("--lamb_l", type=float, default=1.0, help="weight for the lattice part relative to fc")

"""
G       空间群
W       Wyckoff 位置
A       原子类型
XYZ     分数坐标
L       晶格参数"""

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms') #padding用来把所有所有晶体结构补齐到同一个长度
group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

group = parser.add_argument_group('sampling parameters')
group.add_argument('--seed', type=int, default=None, help='random seed to sample')
group.add_argument('--wyckoff', type=str, default=None, nargs='+', help='The Wyckoff positions to be sampled, e.g. a, b')
group.add_argument('--formula', type=str, default=None, help='chemical formula of the compound')
group.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
group.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')
group.add_argument('--K', type=int, default=30, help='top K number of space groups. 0 means we sample spacegroup')
group.add_argument('--spacegroup', type=int, default=None, help='the spacegroup number 1-230, given that will overwrites K')
group.add_argument('--num_samples', type=int, default=10, help='number of generated samples')
group.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io') #并行处理读写的进程的数目
group.add_argument('--save_path', type=str, default=None, help='path to save the sampled structures')
group.add_argument('--output_filename', type=str, default='output.csv', help='outfile to save sampled structures')
group.add_argument('--verbose', type=int, default=0, help='verbose level')
group.add_argument('--remove_radioactive', action='store_true', help='remove radioactive elements and noble gas, only valid when formula is None')


args = parser.parse_args()

args.folder = resolve_path(args.folder)
args.restore_path = resolve_path(args.restore_path)
args.train_path = resolve_path(args.train_path)
args.valid_path = resolve_path(args.valid_path)
args.test_path = resolve_path(args.test_path)
args.save_path = resolve_path(args.save_path)

if args.batchsize <= 0:
    raise ValueError("--batchsize must be a positive integer")
if args.epochs < 0:
    raise ValueError("--epochs must be non-negative")
if args.val_interval <= 0:
    raise ValueError("--val_interval must be a positive integer")
if args.num_io_process <= 0:
    raise ValueError("--num_io_process must be a positive integer")
if args.num_samples < 0:
    raise ValueError("--num_samples must be non-negative")
if args.output_filename.count(".") == 0:
    raise ValueError("--output_filename must include a file extension, for example output.csv")

configure_jax_platform(args.platform)

key = jax.random.PRNGKey(42)

num_cpu = multiprocessing.cpu_count()
print('number of available cpu: ', num_cpu)
if args.num_io_process > num_cpu:
    print('num_io_process should not exceed number of available cpu, reset to ', num_cpu)
    args.num_io_process = num_cpu


################### Data #############################
if args.optimizer != "none":
    train_data = GLXYZAW_from_file(args.train_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
    valid_data = GLXYZAW_from_file(args.valid_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
else:
    # test_data = GLXYZAW_from_file(args.test_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)

    if args.remove_radioactive:
        from crystalformer.src.elements import radioactive_elements_dict, noble_gas_dict
        # remove radioactive elements and noble gas
        atom_mask = [1] + [1 if i not in radioactive_elements_dict.values() and i not in noble_gas_dict.values() else 0 for i in range(1, args.atom_types)]
        atom_mask = jnp.array(atom_mask)
        print('sampling structure formed by non-radioactive elements and non-noble gas')
        
    else:
        atom_mask = jnp.ones((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
    print(atom_mask)
    
    if args.wyckoff is not None:
        idx = [letter_to_number(w) for w in args.wyckoff]
        if any(w is None for w in idx):
            raise ValueError("--wyckoff values must be letters from a-z or A")
        if len(idx) > args.n_max:
            raise ValueError("--wyckoff cannot contain more entries than --n_max")
        # padding 0 until the length is args.n_max
        w_mask = idx + [0]*(args.n_max -len(idx))
        # w_mask = [1 if w in idx else 0 for w in range(1, args.wyck_types+1)]
        w_mask = jnp.array(w_mask, dtype=int)
        print ('sampling structure formed by these Wyckoff positions:', args.wyckoff)
        print (w_mask)
    else:
        w_mask = None

################### Model #############################
params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                      args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, args.embed_size, 
                                      args.atom_types, args.wyck_types,
                                      args.dropout_rate, args.attn_dropout)
transformer_name = 'Nf_%d_Kx_%d_Kl_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_e_%d_drop_%g_%g'%(args.Nf, args.Kx, args.Kl, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size, args.embed_size, args.dropout_rate, args.attn_dropout)

print ("# of transformer params", ravel_pytree(params)[0].size) #把pytree的结构给展平，返回两个值，第一个是展平之后的结果，第二个是用来回复pytree结构的一个callable的向量

################### Train #############################

loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer, args.lamb_a, args.lamb_w, args.lamb_l)

print("\n========== Prepare logs ==========")
if args.optimizer != "none":
    run_name = args.optimizer+"_cfg_%g_bs_%d_lr_%g_decay_%g_clip_%g" % (args.cfg_drop_prob, args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                   + '_A_%g_W_%g_N_%g'%(args.atom_types, args.wyck_types, args.n_max) \
                   + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                   + ('_a_%g_w_%g_l_%g'%(args.lamb_a, args.lamb_w, args.lamb_l)) \
                   +  "_" + transformer_name 
    output_path = os.path.join(args.folder, run_name)

    os.makedirs(output_path, exist_ok=True)
    print("Create directory for output: %s" % output_path)
else:
    output_path = sampling_output_path(args.save_path, args.restore_path)
    os.makedirs(output_path, exist_ok=True)
    print("Will output samples to: %s" % output_path)


print("\n========== Load checkpoint==========")
ckpt = None
try:
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or output_path) 
except FileNotFoundError:
    ckpt_filename, epoch_finished = None, 0
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

if args.optimizer != "none":

    schedule = lambda t: args.lr/(1+args.lr_decay*t)

    if args.optimizer == "adam":
        optimizer = optax.chain(optax.clip_by_global_norm(args.clip_grad), #全局裁剪梯度
                                optax.scale_by_adam(),  #运用adam的算法
                                optax.scale_by_schedule(schedule),  #乘以学习率
                                optax.scale(-1.)) #反一个方向
    elif args.optimizer == 'adamw':
        optimizer = optax.chain(optax.clip(args.clip_grad),
                                optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
                               )

    opt_state = optimizer.init(params)
    if ckpt is not None:
        opt_state = ckpt["opt_state"]
 
    print("\n========== Start training ==========")
    params, opt_state = train(key, optimizer, opt_state, loss_fn, params, epoch_finished, args.epochs, args.batchsize, train_data, valid_data, output_path, args.val_interval, args.cfg_drop_prob)

else:

    print("\n========== Start sampling ==========")
    # jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice 
    #FYI, the error was [Compiling module extracted] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.

    if args.formula is not None:
        composition = formula_string_to_composition_vector(args.formula)
        print ('composition vector of', args.formula)
    else:
        composition = jnp.zeros(args.atom_types)
    print (composition)
    if args.spacegroup is None: 
        if args.K >0:
            print (f'sample from top {args.K} spacegroups')
        else:
            print ('sample spacegroup with temperature', args.temperature)
    else:
        print ('targeting spacegroup No.', args.spacegroup)

    sample_crystal = make_sample_crystal(transformer, args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, w_mask, args.top_p, args.temperature, args.K, args.spacegroup, atom_mask)
    # 这里返回是jitwrapped，是一个可以被jit编译的静态的对象，提高采样速度
    if args.seed is not None:
        key = jax.random.PRNGKey(args.seed) # reset key for sampling if seed is provided

    num_batches = math.ceil(args.num_samples / args.batchsize)

    name, extension = args.output_filename.rsplit('.', 1)
    filename = os.path.join(
        output_path,
        f"{name}{'' if args.formula is None else f'_{args.formula}'}.{extension}"
    )
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, args.num_samples) 
        n_sample = end_idx - start_idx
        key, subkey = jax.random.split(key)
        G, XYZ, A, W, M, L = sample_crystal(subkey, params, n_sample, composition)
        
        if args.verbose>1:
            print ("G:\n", G)  # spacegroup
            print ("XYZ:\n", XYZ)  # fractional coordinate 
            print ("A:\n", A)  # element type
            print ("W:\n")  # Wyckoff positions
            for (g, w) in zip(G, W):
                print (g, [_w.item() for _w in w])
            print ("M:\n", M)  # multiplicity 
            print ("N:\n", M.sum(axis=-1)) # total number of atoms
            print ("L:\n")  # lattice
            for g, l in zip(G, L):
                print (g, l)
            for g, a in zip(G, A):
                print(g, [element_list[i] for i in a])

        # output G, L, X, A, W, M, AW to csv file
        # output logp_g, logp_w, logp_xyz, logp_a, logp_l to csv file
        data = pd.DataFrame()
        data['G'] = np.array(G).tolist()
        data['L'] = np.array(L).tolist()
        data['X'] = np.array(XYZ).tolist()
        data['A'] = np.array(A).tolist()
        data['W'] = np.array(W).tolist()
        data['M'] = np.array(M).tolist()

        num_atoms = jnp.sum(M, axis=1)
        length, angle = jnp.split(L, 2, axis=-1)
        length = length/num_atoms[:, None]**(1/3)
        angle = angle * (jnp.pi / 180) # to rad
        L = jnp.concatenate([length, angle], axis=-1)

        if args.verbose>1:
            print ("reduced L:\n")  # lattice
            for g, l in zip(G, L):
                print (g, l)

        # Repeat composition to match the actual sampled batch size.
        composition_batch = composition[None, :].repeat(n_sample, axis=0) #composition是119个元素的one-hot标记，这里先把他变成一个向量，再堆叠
        logp_g, logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=8)(params, key, composition_batch, G, L, XYZ, A, W, False)

        data['logp_g'] = np.array(logp_g).tolist()
        data['logp_w'] = np.array(logp_w).tolist()
        data['logp_xyz'] = np.array(logp_xyz).tolist()
        data['logp_a'] = np.array(logp_a).tolist()
        data['logp_l'] = np.array(logp_l).tolist()

        sample_logp = logp_g + logp_xyz + args.lamb_w*logp_w + args.lamb_a*logp_a + args.lamb_l*logp_l
        data['logp'] = np.array(sample_logp).tolist()

        actual_compositions = jax.vmap(find_composition_vector)(A, M) #actual同样是长度为119，只不过取了一个GCD化成了最简形式
        formula_match = jnp.all(actual_compositions == composition_batch, axis=1) #actual_compositions和composition_batch是两个长度为119的向量，formula_match是长度为batchsize的布尔向量，表示每个样本的实际化学式是否和目标化学式一致

        if args.verbose>0:
            idx = jnp.argsort(G[formula_match])
            print ('sample logp of matched formula:', sample_logp[formula_match][idx])
            print ('G[formula_match]:\n', G[formula_match][idx])
            print ('W[formula_match]:\n', W[formula_match][idx])
            print ('A[formula_match]:\n', A[formula_match][idx])

        data = data.sort_values(by='G', ascending=True) 
        
        # Use write mode for first batch, append mode for subsequent batches
        write_mode = 'w' if batch_idx == 0 else 'a'
        write_header = True if batch_idx == 0 else False
        data.to_csv(filename, mode=write_mode, index=False, header=write_header)

        print ("Wrote samples to %s (batch %d/%d)"%(filename, batch_idx + 1, num_batches))
