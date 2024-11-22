"""Module loads mass assembly history data for diffmah
"""

import numpy as np

try:
    from dsps.cosmology import flat_wcdm

    HAS_DSPS = True
except ImportError:
    HAS_DSPS = False

try:
    from haccytrees import Simulation as HACCSim
    from haccytrees import coretrees

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    MPI = COMM = None


MASS_COLNAME = "infall_tree_node_mass"


def _scatter_nd(array, axis=0, comm=COMM, root=0):
    """Scatter n-dimensional array from root to all ranks

    This function is taken from https://github.com/AlanPearl/diffopt

    """
    ans: np.ndarray = np.array([])
    if comm.rank == root:
        splits = np.array_split(array, comm.size, axis=axis)
        for i in range(comm.size):
            if i == root:
                ans = splits[i]
            else:
                comm.send(splits[i], dest=i)
    else:
        ans = comm.recv(source=root)
    return ans


def _load_forest(fn_data, sim_name, chunknum, nchunks):
    if not HAS_HACCYTREES:
        raise ImportError("Must have haccytrees installed to use this data loader")

    try:
        sim = HACCSim.simulations[sim_name]
    except KeyError:
        sim = HACCSim.parse_config(sim_name)

    zarr = sim.step2z(np.array(sim.cosmotools_steps))

    forest_matrices = coretrees.corematrix_reader(
        fn_data,
        calculate_secondary_host_row=True,
        nchunks=nchunks,
        chunknum=chunknum,
        simulation=sim,
    )
    return sim, forest_matrices, zarr


def load_mahs(fn_data, sim_name, chunknum, nchunks, mass_colname=MASS_COLNAME):

    sim, forest_matrices, zarr = _load_forest(fn_data, sim_name, chunknum, nchunks)
    mahs = forest_matrices[mass_colname]

    # Ensure the target MAHs are cumulative peak masses
    mahs = np.maximum.accumulate(mahs, axis=1)

    if not HAS_DSPS:
        raise ImportError("Must have dsps installed to use this data loader")

    cosmo_dsps = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )

    tarr = flat_wcdm.age_at_z(zarr, *cosmo_dsps)

    return tarr, mahs


def load_mahs_per_rank(
    fn_data, sim_name, chunknum, nchunks, mass_colname=MASS_COLNAME, comm=None
):
    if comm is None:
        comm = MPI.COMM_WORLD

    if comm.rank == 0:
        tarr, mahs = load_mahs(
            fn_data, sim_name, chunknum, nchunks, mass_colname=mass_colname
        )
    else:
        tarr = None
        mahs = None
    mahs_for_rank = _scatter_nd(mahs, axis=0, comm=comm, root=0)
    tarr = comm.bcast(tarr, root=0)

    return tarr, mahs_for_rank
