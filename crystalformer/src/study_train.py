import jax
import jax.numpy as jnp
from functools import partial
import os
import optax

from crystalformer.src.utils import shuffle
import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.formula import find_composition_vector
from crystalformer.src.wyckoff import mult_table


shard = jax.pmap(lambda x: x)
p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))

    
def scatter(x: jnp.ndarray, retain_axis=False) -> jnp.ndarray:
    num_devices = jax.local_device_count()
    if x.shape[0] % num_devices != 0:
        raise ValueError("The first dimension of x must be divisible by the total number of GPU devices. "
                         "Got x.shape[0] = %d for %d devices now." % (x.shape[0], num_devices))
    dim_per_device = x.shape[0] // num_devices
    x = x.reshape(
        (num_devices,) +
        (() if dim_per_device == 1 and not retain_axis else (dim_per_device,)) +
        x.shape[1:]
    )
    return shard(x)


@jax.jit
def add_composition_with_cfg_drop(key, data, cfg_drop_prob):
    # compute composition vector and apply classifer-free guidance training
    G, A, W = data[0], data[3], data[4]
    M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # M是(batchsize, n_max)   这里的G是(batchsize,) W是(batchisize, n_max)
    composition = jax.vmap(find_composition_vector, (0, 0), 0)(A, M) #(batchisize, 118)
    # drop composition with probability cfg_drop_prob, set to zero vector
    drop_mask = jax.random.uniform(key, (composition.shape[0],)) < cfg_drop_prob
    composition = jnp.where(drop_mask[:, None], jnp.zeros_like(composition), composition)
    data = (composition,) + data  # 也就是把composition接在这个元组的最前面

    return data


def train(key, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path, val_interval, cfg_drop_prob):
    num_devices = jax.local_device_count()
    train_samples = train_data[1].shape[0] #train_data和valid_data都是(G, L, XYZ, A, W) 元组，且GLXYZAW都是（N,……）
    valid_samples = valid_data[1].shape[0] 
    max_batchsize = min(batchsize, train_samples, valid_samples)
    batchsize = (max_batchsize // num_devices) * num_devices
    if batchsize == 0:
        raise ValueError(
            "Not enough samples for the available devices. "
            f"Got train={train_samples}, valid={valid_samples}, devices={num_devices}."
        )
    batch_per_device = batchsize // num_devices
    shape_prefix = (num_devices, batch_per_device)
    print("num_devices: ", num_devices)
    print("effective_batchsize: ", batchsize) #effective_batchsize是我们真实训练用的batchsize
    print("batch_per_device: ", batch_per_device) 
    print("shape_prefix: ", shape_prefix)

    key = jax.random.fold_in(key, jax.process_index())  # 把process_index编入我们的key，这样不同的process就有不同的key
    key, *keys = jax.random.split(key, num_devices + 1)
    keys = scatter(jnp.array(keys))

    @partial(jax.pmap, axis_name="p", in_axes=(None, 0, None, 0), out_axes=(None, None, 0),)
    def update(params, key, opt_state, data):
        composition, G, L, X, A, W = data
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, key, composition, G, L, X, A, W, True) 
        #这里的value_and_grad返回的是一个函数，jax.value_and_grad(fun, argnums=0, has_aux=False)(*args)，默认对第1个参数求导，后面第二个括号的参数是传给在第一个括号被传入的函数的，所谓辅助数据是loss_fn的返回值的第二项
        grad = jax.lax.pmean(grad, axis_name="p")
        value = jax.lax.pmean(value, axis_name="p")
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n") #"w" 覆盖写入， "a" 追加写入；buffering=1 表示按行刷新，newline="\n" 指定换行符，f 是文件对象。
    if os.path.getsize(log_filename) == 0:
        f.write("epoch t_loss v_loss t_loss_g v_loss_g t_loss_w v_loss_w t_loss_a v_loss_a t_loss_xyz v_loss_xyz t_loss_l v_loss_l\n")
    #如果训练日志文件是空的就先写一个表头
 
    for epoch in range(epoch_finished+1, epochs+1):
        key, subkey = jax.random.split(key)
        train_data = shuffle(subkey, train_data) #打乱所有训练样本

        _, train_L, _, _, _ = train_data # train_data是G,L,XYZ,A,W组成的元组

        train_loss = 0.0 
        train_aux = 0.0, 0.0, 0.0, 0.0, 0.0
        num_samples = train_L.shape[0]
        num_batches = num_samples // batchsize
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = start_idx + batchsize
            data = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], train_data) #取出我们这个轮次要处理的数据
            key, subkey = jax.random.split(key)
            data = add_composition_with_cfg_drop(subkey, data, cfg_drop_prob)
            data = jax.tree_util.tree_map(lambda x: x.reshape(shape_prefix + x.shape[1:]), data) #开始把数据分给多个设备，把最前面的batchsize拆成设备数*每个设备的样本数，每个batch都会分给多个设备
            
            keys, subkeys = p_split(keys)
            params, opt_state, (loss, aux) = update(params, subkeys, opt_state, data)
            train_loss, train_aux = jax.tree_util.tree_map(   
                        lambda acc, i: acc + jnp.mean(i),
                        (train_loss, train_aux),  
                        (loss, aux)
                        )

        train_loss, train_aux = jax.tree_util.tree_map(
                        lambda x: x/num_batches, 
                        (train_loss, train_aux)
                        ) #后面没用了，再用的时候会先初始化，没别的特殊含义，为了方便日志的输出

        if epoch % val_interval == 0: #如果到了valid轮次，其实还是进行相同的操作，只不过要用valid库里面的数据检验一下训练的成果
            _, valid_L, _, _, _ = valid_data 
            valid_loss = 0.0 
            valid_aux = 0.0, 0.0, 0.0, 0.0, 0.0
            num_samples = valid_L.shape[0]
            num_batches = num_samples // batchsize
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = start_idx + batchsize
                data = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], valid_data)
                key, subkey = jax.random.split(key)
                data = add_composition_with_cfg_drop(subkey, data, cfg_drop_prob)
                data = jax.tree_util.tree_map(lambda x: x.reshape(shape_prefix + x.shape[1:]), data)

                keys, subkeys = p_split(keys)
                loss, aux = jax.pmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0),
                                     static_broadcasted_argnums=8)(params, subkeys, *data, False)
                valid_loss, valid_aux = jax.tree_util.tree_map(
                        lambda acc, i: acc + jnp.mean(i),
                        (valid_loss, valid_aux), 
                        (loss, aux)
                        )

            valid_loss, valid_aux = jax.tree_util.tree_map(
                        lambda x: x/num_batches, 
                        (valid_loss, valid_aux)
                        ) 

            train_loss_g, train_loss_w, train_loss_a, train_loss_xyz, train_loss_l = train_aux
            valid_loss_g, valid_loss_w, valid_loss_a, valid_loss_xyz, valid_loss_l = valid_aux

            f.write( ("%6d" + 12*"  %.6f" + "\n") % (epoch, 
                                                    train_loss,   valid_loss,
                                                    train_loss_g, valid_loss_g, 
                                                    train_loss_w, valid_loss_w, 
                                                    train_loss_a, valid_loss_a, 
                                                    train_loss_xyz, valid_loss_xyz, 
                                                    train_loss_l, valid_loss_l
                                                    ))

            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            if jax.process_index() == 0:
                checkpoint.save_data(ckpt, ckpt_filename)
                print("Save checkpoint file: %s" % ckpt_filename)
                
    f.close()
    return params, opt_state
