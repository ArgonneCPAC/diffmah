"""
"""

import numpy as np
from dsps.cosmology import flat_wcdm
from haccytrees import Simulation as HACCSim
from haccytrees import coretrees

from mpi4py import MPI
from diffopt.multigrad.util import scatter_nd

import desi_cosmo

TASSO_DRN_DESI = "/Users/aphearin/work/DATA/DESI_W0WA"
MASS_COLNAME = "infall_tree_node_mass"

NCHUNKS = 20
NUM_SUBVOLS_DISCOVERY = 96


def load_forest(fn_data, fn_cfg, chunknum, nchunks=NCHUNKS):
    sim = HACCSim.parse_config(fn_cfg)
    zarr = sim.step2z(np.array(sim.cosmotools_steps))

    forest_matrices = coretrees.corematrix_reader(
        fn_data,
        calculate_secondary_host_row=True,
        nchunks=nchunks,
        chunknum=chunknum,
        simulation=sim,
    )
    return sim, forest_matrices, zarr


def load_mahs(fn_data, fn_cfg, chunknum, nchunks=NCHUNKS, mass_colname=MASS_COLNAME):

    sim, forest_matrices, zarr = load_forest(fn_data, fn_cfg, chunknum, nchunks=nchunks)
    mahs = forest_matrices[mass_colname]

    if sim.name == "DESI_LCDM":
        cosmo_params = desi_cosmo.dlc_params
    elif sim.name == "DESI_W0WA":
        cosmo_params = desi_cosmo.dwc_params
    else:
        raise ValueError("Unrecognized simulation name")
    tarr = flat_wcdm.age_at_z(zarr, *cosmo_params)

    return tarr, mahs


def load_mahs_per_rank(fn_data, fn_cfg, chunknum, nchunks=NCHUNKS,
                       mass_colname=MASS_COLNAME, comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD

    if comm.rank == 0:
        tarr, mahs = load_mahs(
            fn_data, fn_cfg, chunknum, nchunks=nchunks,
            mass_colname=mass_colname
        )

        # Ensure the target MAHs are cumulative peak masses
        mahs = np.maximum.accumulate(mahs, axis=1)
    else:
        tarr = None
        mahs = None
    mahs_for_rank = scatter_nd(mahs, axis=0, comm=comm, root=0)
    tarr = comm.bcast(tarr, root=0)

    if comm.rank == 0:
        print("Number of halos in chunk = {}".format(mahs.shape[0]))

    return tarr, mahs_for_rank
