import os
import numpy as np
from pymatgen.core import Structure, Lattice

_symops = None
_mult_table = None
_wmax_table = None


def _symmetry_tables():
    """Load Wyckoff tables only when a generated structure is evaluated."""

    global _symops, _mult_table, _wmax_table
    if _symops is None:
        from crystalformer.src.wyckoff import mult_table, symops, wmax_table

        _symops = np.asarray(symops)
        _mult_table = np.asarray(mult_table)
        _wmax_table = np.asarray(wmax_table)
    return _symops, _mult_table, _wmax_table


def __getattr__(name):
    """Lazily preserve the legacy symmetry-table module attributes."""

    table_names = {"symops": 0, "mult_table": 1, "wmax_table": 2}
    if name in table_names:
        return _symmetry_tables()[table_names[name]]
    raise AttributeError(name)

def all_nonzero_gpus():
    import torch

    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(1, n)]

def relax_structures(relaxer, atoms_list):
    """
    Args:
        relaxer: BatchRelaxer object
        structures: List of ASE atoms 

    Returns:
        final_structures: List of final structures
        final_energies: List of final energies

    """

    from pymatgen.io.ase import AseAtomsAdaptor

    ase_adaptor = AseAtomsAdaptor()

    relaxation_trajectories = relaxer.relax(atoms_list)

    # Extract the relaxed structures and corresponding energies
    final_structures = [traj[-1] for traj in relaxation_trajectories.values()]
    final_energies = [structure.info['total_energy'] for structure in final_structures]
    final_structures = [ase_adaptor.get_structure(struct) for struct in final_structures]
    return final_structures, final_energies

def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops 
    we need to do that because the sampled atom might not be at the first WP
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''

    symops, mult_table, wmax_table = _symmetry_tables()

    # (1) apply all space group symmetry op to the x 
    w_max = wmax_table[g-1].item()
    m_max = mult_table[g-1, w_max].item()
    ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= np.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = np.dot(symops[g-1, w, 0], np.array([*coord, 1])) - coord
        diff -= np.rint(diff)
        return np.sum(diff**2) 
   #  loc = np.argmin(jax.vmap(dist_to_op0x)(coords))
    loc = np.argmin([dist_to_op0x(coord) for coord in coords])
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= np.floor(xs) # wrap back to 0-1 
    return xs


def get_atoms_from_GLXYZAW(G, L, XYZ, A, W):
    # Convert once at the boundary and derive every padding mask from the
    # original atom sequence.  The old implementation filtered ``A`` first
    # and then reused indices from the shortened array for ``XYZ``/``W``;
    # that silently selected the wrong sites when padding was interspersed.
    A = np.asarray(A).reshape(-1)
    XYZ = np.asarray(XYZ)
    W = np.asarray(W).reshape(-1)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must have shape (n_sites, 3)")
    if not (A.size == XYZ.shape[0] == W.size):
        raise ValueError("XYZ, A and W must have the same number of sites")
    active = A != 0
    A = A[active]
    X = XYZ[active]
    W = W[active]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    A_list = np.repeat(A, [len(xs) for xs in xs_list])
    X_list = np.concatenate(xs_list)
    struct = Structure(lattice, A_list, X_list)
    struct = struct.get_primitive_structure().to_ase_atoms()
    return struct


def get_structure_from_GLXYZAW(G, L, XYZ, A, W):
    """Return a pymatgen structure without importing the MLFF stack.

    XRD and other CPU-only rewards should use this helper instead of creating
    ASE atoms.  The implementation lives in ``reinforce.xrd`` so importing
    this module does not initialize ORB or BatchRelaxer.
    """

    from crystalformer.reinforce.xrd import structure_from_GLXYZAW

    return structure_from_GLXYZAW(G, L, XYZ, A, W)


def make_ehull_reward_fn(calculator, ref_data, batch=50, n_jobs=-1, relaxation=False, clip_value=10.0):
    """
    Args:
        calculator: ase calculator object
        ref_data: reference data for ehull calculation

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """

    from crystalformer.reinforce import ehull
    import joblib
    import pandas as pd
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from BatchRelaxer import BatchRelaxer
    from pymatgen.io.ase import AseAtomsAdaptor

    ase_adaptor = AseAtomsAdaptor()

    def energy_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            atoms.calc = calculator
            #relax the structure before evaluating ehull
            optimizer = FrechetCellFilter(atoms)
            FIRE(optimizer).run(fmax=0.01, steps=500)  
            energy = atoms.get_potential_energy()
            structure = ase_adaptor.get_structure(atoms)
        except:
            energy = np.inf
            structure = None
        
        return structure, energy

    def reward_fn(structure, energy):
        if structure == None:
            e_above_hull = np.inf
        else:    
            try: 
                e_above_hull = ehull.forward_fn(structure, energy, ref_data)
            except:
                e_above_hull = np.inf

        # clip e above hull to avoid too large or too small values
        e_above_hull = np.clip(e_above_hull, -clip_value, clip_value)

        return e_above_hull
    
    def map_reward_fn(structures, energies):
        output = map(reward_fn, structures, energies)

        return list(output)
    
    def parallel_reward_fn(structures, energies):
        xs = [(structures[i:i+batch], energies[i:i+batch]) for i in range(0, len(structures), batch)]
        output = joblib.Parallel(
                        n_jobs=n_jobs
                    )(joblib.delayed(map_reward_fn)(*x) for x in xs)
        # concatenate the output
        output = np.concatenate(output)

        return output

    def batch_reward_fn(x, path=None, epoch=None):
        import jax
        import jax.numpy as jnp

        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)

        x = (G, L, XYZ, A, W)
        
        #series relax
        #structures, energies = zip(*map(energy_fn, zip(*x)))

        #batch relax 
        steps = 500 if relaxation else 0
        atoms_list = [get_atoms_from_GLXYZAW(*t) for t in zip(*x)]
        
        if jax.local_device_count() > 1:
            structures, energies = BatchRelaxer.relax_multi_gpu(
                        potential=calculator.model,
                        atoms_list=atoms_list,
                        max_natoms_per_batch=1000, 
                        fmax=0.01,
                        max_n_steps=steps,
                        filter="FRECHETCELLFILTER",
                        optimizer="FIRE"
                    )
        else: 
            relaxer = BatchRelaxer(calculator.model,
                               device='cuda',
                               fmax=0.01,
                               max_n_steps=steps, 
                               max_natoms_per_batch=1000,
                               filter="FRECHETCELLFILTER",
                               optimizer="FIRE")
            structures, energies = relax_structures(relaxer, atoms_list)
        
        output = map_reward_fn(structures, energies)
        output = jnp.array(output)
        output = jax.device_put(output, jax.devices()[0]).block_until_ready()

        if path is not None and epoch is not None:
            data = pd.DataFrame()
            data['relaxed_ehull'] = output
            relaxed_cif_dicts = []
            for structure in structures:
                structure = structure.remove_site_property("forces")
                structure.properties = None
                relaxed_cif_dicts.append(structure.as_dict())
            data['relaxed_cif'] = relaxed_cif_dicts
            data.to_csv(os.path.join(path, f"relaxed_structures_ehull_{epoch}.csv"),
                            index=False)

        return output  

    return reward_fn, batch_reward_fn


def make_xrd_reward_fn(*args, **kwargs):
    """Lazy public wrapper for the CPU-only XRD reward factory."""

    from crystalformer.reinforce.xrd import make_xrd_reward_fn as _make_xrd_reward_fn

    return _make_xrd_reward_fn(*args, **kwargs)


def make_prop_reward_fn(model, target, dummy_value=5, loss_type='mse'):

    """
    Args:
        model: property prediction model, takes pymatgen structure as input, returns property value
        target: target property value
        dummy_value: dummy value to return if model fails to predict
        loss_type: loss function type, 'mse' or 'mae'

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function

    """

    from pymatgen.io.ase import AseAtomsAdaptor

    ase_adaptor = AseAtomsAdaptor()

    def reward_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            struct = ase_adaptor.get_structure(atoms)
            quantity = model(struct)
            # if quantity is nan, return a dummy value
            quantity = quantity if not np.isnan(quantity) else np.array(dummy_value)
        except:
            quantity = np.array(dummy_value)  #TODO: check if this is a good idea
        
        return quantity

    def batch_reward_fn(x, path=None, epoch=None):
        import jax
        import jax.numpy as jnp

        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = jnp.array(list(output)) - target
        
        if loss_type == 'mae':
            output = jnp.abs(output)
        elif loss_type == 'mse':
            output = output**2  # MSE loss
        else:
            raise ValueError('Invalid loss type')
        
        output = jax.device_put(output, jax.devices()[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn


def make_dielectric_reward_fn(models, dummy_value=0):
    """
    Reward function for dielectric reward. models contains two models, one for dielectric constant and one for band gap.
    the reward is the product of the two quantities.

    Args:
        models: list of property prediction models, each takes pymatgen structure as input, returns property value
        dummy_value: dummy value to return if model fails to predict

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """

    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.stress import voigt_6_to_full_3x3_stress
    ase_adaptor = AseAtomsAdaptor()

    assert len(models) == 2, 'models should contain two models, one for dielectric constant and one for band gap'

    def reward_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            struct = ase_adaptor.get_structure(atoms)
            
            pred = models[0](struct)
            dielectric_tensor = voigt_6_to_full_3x3_stress(pred)
            eigenvalues, _ = np.linalg.eig(dielectric_tensor)
            scalar_dielectric = np.mean(np.real(eigenvalues))
            if np.isnan(scalar_dielectric):
                return np.array(dummy_value)

            band_gap = models[1](struct).item()
            if np.isnan(band_gap):
                return np.array(dummy_value)
            
            reward = - np.array(scalar_dielectric * band_gap)

        except:
            reward = np.array(dummy_value)  #TODO: check if this is a good idea
        
        return reward


    def batch_reward_fn(x, path=None, epoch=None):
        import jax
        import jax.numpy as jnp

        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = np.array(list(output))
        output = jax.device_put(output, jax.devices()[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn


def make_density_reward_fn(inverse=False):
    """
    Reward function for density reward.

    Args:
        inverse: if True, return negative density as reward

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """

    from pymatgen.io.ase import AseAtomsAdaptor
    ase_adaptor = AseAtomsAdaptor()

    def reward_fn(x):
        G, L, XYZ, A, W = x
        atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
        struct = ase_adaptor.get_structure(atoms)

        return struct.density if not inverse else -struct.density

    def batch_reward_fn(x, path=None, epoch=None):
        import jax
        import jax.numpy as jnp

        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = np.array(list(output))
        output = jax.device_put(output, jax.devices()[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn
