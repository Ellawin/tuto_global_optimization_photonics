# %% [markdown]
# # Tutorial: Plasmonic directional nanoantenna optimization - MPI-run
# 
# Here we use [`pyGDM`](https://homepages.laas.fr/pwiecha/pygdm_doc) together with [`nevergrad`](https://facebookresearch.github.io/nevergrad/) to optimize a plasmonic nanoantenna for directional emission of an attached quantum emitter.
# The example is an MPI-parallelized version to accelerate the benchmarking of multiple algorithms.
#
# To start its execution, mpi and mpi4py need to be installed. 
# Then, for example to run 8 parallel processes, launch via:
# "mpirun -n 8 python python_script.py"
# 
# This tutorial aims at reproducing the generat trend in [Wiecha et al. Opt. Express 27, pp. 29069, (2019)](https://doi.org/10.1364/OE.27.029069). However, for runtime and readibility of the notebook, we keep the configuration simpler. The nanoantenna is made from smaller and fewer gold elements, the antenna is placed in a homogeneous environment (air) and the emitter is *above* the antenna, such that we do not need to add a forbidden zone constraint in the material positioning.

# %% [markdown]
# ## load modules

# %%
import nevergrad as ng
import numpy as np
from tqdm import tqdm
import warnings


# limit CPU cores per process
from threadpoolctl import threadpool_limits
threadpool_limits(limits=2, user_api='blas')
import numba
numba.set_num_threads(2)


# multiprocessing for parallelized optimization runs
import pickle

from pyGDM2 import structures
from pyGDM2 import materials
from pyGDM2 import fields
from pyGDM2 import core
from pyGDM2 import propagators
from pyGDM2 import linear
from pyGDM2 import tools
from pyGDM2 import visu

# mpi config
from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs, rank = comm.Get_size(), comm.Get_rank()

## --- create list of jobs and split in equal parts depending on `nprocs`
def split(jobs, nprocs):
    return [jobs[i::nprocs] for i in range(nprocs)]


# %% [markdown]
# ## configure the optimization
# 
# Configure the number of plasmonic blocks and the position constraints in the nevergrad instrumentation

# %%
# configure the parametrization
N_elements = 40
min_pos = -12
max_pos = 12
method = 'lu'  # simulations method: 'lu' --> CPU, 'cupy' --> GPU (Nvidia CUDA)


# configure multiple optimization runs
budget = 10000       # stop criterion: allowed number of evaluations
N_repet_each = 10   # how often to repeat the run with each optimizer

# list of algorithms to use
list_optims = ["TwoPointsDE", "BFGS", 
               "MultiDiscrete", "DoubleFastGADiscreteOnePlusOne", 
               "PSO", "CMA", "QOPSO", "NGOpt"]




# %% [markdown]
# ## preparation define geometry parametrization & cost functions
# 
# We start by defining the geometric model helper and the cost function

# %%
def setup_structure(XY_coords_blocks, element_sim):
    """helper to create structure, from positions of gold elements
    each positions in units of discretization steps

    Args:
        XY_coords_blocks (list): list gold element positions (x1,x2,x3,...,y1,y2,....)
        element_sim (`pyGDM2.core.simulation`): single element simulation
        
    Returns:
        pyGDM2.structures.struct: instance of nano-geometry class
    """
    
    n = len(XY_coords_blocks) // 2
    x_list = XY_coords_blocks[:n]
    y_list = XY_coords_blocks[n:]
    pos = np.transpose([x_list, y_list])
    
    struct_list = []
    for _p in pos:
        x, y = _p
        # displace by steps of elementary block-size
        _s = element_sim.struct.copy()
        DX = _s.geometry[:, 0].max() - _s.geometry[:, 0].min() + _s.step
        DY = _s.geometry[:, 1].max() - _s.geometry[:, 1].min() + _s.step
        _s = structures.shift(_s, np.array([DX*int(x), DY*int(y), 0.0]))
        
        # do not add the block if too close to emitter at (0,0) 
        if np.abs(x) >= 1 or np.abs(y) >= 1:
            struct_list.append(_s)
    
    if len(struct_list) == 0:
        struct_list.append(_s + [DX, DY, 0])  # add at least one block
    
    full_struct = structures.combine_geometries(struct_list, step=element_sim.struct.step)
    full_sim = element_sim.copy()
    full_sim.struct = full_struct
    return full_sim


# ------- the optimization target function -------
def cost_direct_emission(x, element_sim, method, verbose=0):
    """ cost function: maximize scattering towards small solid angle

    Args:
        x (list): optimization params --> pos of elements
        element_sim (`pyGDM2.core.simulation`): single element simulation
        method (str): pyGDM2 solver method

    Returns:
        float: 1 - Reflectivity at target wavelength
    """
    sim = setup_structure(x, element_sim)
    sim.scatter(method=method, verbose=verbose)
    
    ## 2D scattering evaluation in upper hemisphere
    warnings.filterwarnings('ignore')
    Nteta, Nphi = 18, 32
    NtetaW, NphiW = 4, 5
    Delta_angle = np.pi * 10/180   # +/- 10 degrees target angle
    I_full = linear.farfield(
        sim, field_index=0, return_value='int_Etot',
        phimin=0, phimax=2*np.pi,
        tetamin=0, tetamax=np.pi/2,
        Nteta=Nteta, Nphi=Nphi)
    I_window = linear.farfield(
        sim, field_index=0, return_value='int_Etot',
        phimin=-np.pi/6, phimax=np.pi/6 + (np.pi/3)/NphiW,  # supposed to start at zero, excluding last point
        tetamin=np.pi/2 - Delta_angle, tetamax=np.pi/2 + Delta_angle,
        Nteta=NtetaW, Nphi=NphiW)
    
    cost =  -1 * (I_window / I_full)
    if verbose: 
        print('cost: {:.5f}'.format(cost))
    
    return cost

# %% [markdown]
# ## configure the nano-optics problem
# 
# Now we configure the simulation specifics:
# 
#  - the plasmonic building block geometry
#  - the simulation conditions
#  - the illumination

# %%
# ------- define main simulation
## geometry: single small gold rectangle (20x20x10 nm^3)
step = 20
material = materials.gold()
geometry = structures.rect_wire(step, L=2, W=2, H=2)
geometry = structures.center_struct(geometry)
struct = structures.struct(step, geometry, material)

## environment: air
n1 = 1.0
dyads = propagators.DyadsQuasistatic123(n1=n1)


## illumination: local quantum emitter (dipole source)
field_generator = fields.dipole_electric    # light-source: dipolar emitter
kwargs = dict(x0=0, y0=0, z0=step,          # position: center, "steps" above surface
              mx=0, my=1, mz=0,             # orientation: Y
              R_farfield_approx=5000)       # use farfield approx. after 5microns
wavelengths = [800.]
efield = fields.efield(field_generator, wavelengths=wavelengths, kwargs=kwargs)

## simulation object of single element
element_sim = core.simulation(struct, efield, dyads)


# %% [markdown]
# ## Running multiple optimizations in parallel
# 
# Now we run the optimization with several optimizers and several times.
# The positioning parameters are discrete on a fixed grid with steps of the block size. Here we also show how to configure nevergrad for performing a discrete optimization. 
# 
# Because the simulations are relatively costly, we will show here how to use multiprocessing for parallel execution of several optimizaiton runs. Note, that this could be done similarly with python's `multiprocessing`, but pyGDM uses internally openMP and is therefore not python threadsafe. We therefore use `loky` as robust alternative.

# %%
# define a function that run's an optimization
def opt_func(conf):
    optim_name, budget, N_elements, min_pos, max_pos, element_sim, method = conf
    
    # discrete integer parametrization (positions on a grid)
    args_geo_ng = ng.p.Tuple(
        *[ng.p.Scalar(lower=min_pos, upper=max_pos, 
                    # init=np.random.uniform(min_pos, max_pos)  # replace default initialization
                    ).set_integer_casting() 
                for i in range(2*N_elements)]
        )

    # wrap free and fixed arguments
    instru = ng.p.Instrumentation(
        x=args_geo_ng,   # optimized args
        
        element_sim=element_sim, # fixed kwargs
        method=method,
    )
    
    # initialize the optimizer
    optimizer = ng.optimizers.registry[optim_name](instru, budget)
    
    # init tracking values
    best_f = float("inf") # container for best solution
    yval = []             # container for convergence curve
    # the actual optimization loop
    pbar = tqdm(range(budget))
    for k in pbar:
        pbar.set_description("best cost '{}': {:.3f}".format(optim_name, best_f))
        x = optimizer.ask()   # get suggestion for new test structure

        y = cost_direct_emission(**x.value[1]) # eval. the optimizer's suggestion
        optimizer.tell(x, y)  # tell the cost to the optimizer
        
        if y < best_f:
            best_f = y
            best_x = x
        yval.append(best_f)
    
    # finished - return results
    opt_algo = optim_name if optim_name!='NGOpt' else 'NGOpt ({})'.format(optimizer.optim.name)
    return dict(best_f=best_f, 
                best_x=best_x,
                yval=yval,
                optim_name=opt_algo)

# %% [markdown]
# ## configure optimizations and run all in parallel
# config optimizations
conf_list = []
# multiple algos
for optim_name in list_optims:
    # run the optimizer multiple times
    for k in range(N_repet_each):
        conf_list.append([optim_name, budget, N_elements, min_pos, max_pos, element_sim, method])

# %% run in parallel via MPI
if comm.rank == 0:
    jobs = split(conf_list, nprocs)
    if len(np.unique([len(i) for i in jobs])) > 1:
        print("Efficiency warning: Number of wavelengths ({}) ".format(
                                            len(conf_list)) + 
              "not divisable by Nr of processes ({})!".format(nprocs))
else:
    jobs = None

## --- Scatter jobs across processes and perform GDM simulations for each wavelength
jobs = comm.scatter(jobs, root=0)

results_job = []
for job in jobs:
    print(" process #{}: optimizer {}".format(rank, job[0]))
    _res = opt_func(job)
    results_job.append(_res)

## --- Gather results on rank 0 and save
results = MPI.COMM_WORLD.gather(results_job, root=0)
if comm.rank == 0:
    # unwrap results-lists from each process' job-lists
    results = [i for temp in results for i in temp]
    pickle.dump(results, open('results_plasmonics_opt_directional_MPIrun.pkl', 'wb'))

