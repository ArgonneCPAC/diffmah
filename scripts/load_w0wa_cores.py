"""
"""

import numpy as np
from dsps.cosmology import flat_wcdm
from haccytrees import Simulation as HACCSim
from haccytrees import coretrees

import desi_cosmo

TASSO_DRN_DESI = "/Users/aphearin/work/DATA/DESI_W0WA"
MASS_COLNAME = "infall_tree_node_mass"

NCHUNKS = 500
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
