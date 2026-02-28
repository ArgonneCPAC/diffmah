""""""

import os

import numpy as np

try:
    from umachine_pyio.load_mock import load_mock_from_binaries

    HAS_UM_PYIO = True
except ImportError:
    HAS_UM_PYIO = False


try:
    from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm

    HAS_DSPS = True
except ImportError:
    HAS_DSPS = False

BPL_0M0 = 0.307
BPL_h = 0.678
BPL_COSMO = DEFAULT_COSMOLOGY._replace(Om0=BPL_0M0, h=BPL_h, w0=-1.0, wa=0.0)

BPL_MAH_FIT_LGMIN = 10.0
BPL_T_FIT_MIN = 1.0


def load_bpl_mah_data(drn, subvolumes, galprops=("halo_id", "mpeak_history_main_prog")):
    assert HAS_UM_PYIO, "Must have umachine_pyio to use this function"
    mah_data = load_mock_from_binaries(subvolumes, drn, galprops=galprops)

    mah_data["mpeak_history_main_prog"] = np.maximum.accumulate(
        mah_data["mpeak_history_main_prog"], axis=1
    )

    mah_data["mpeak_history_main_prog"] = np.clip(
        mah_data["mpeak_history_main_prog"], a_min=1.0, a_max=np.inf
    )

    assert HAS_DSPS, "Must have dsps to use this function"
    scale_list = np.loadtxt(os.path.join(drn, "scale_list_bpl.dat"))
    redshift_list = 1.0 / scale_list - 1.0
    bpl_t_table = flat_wcdm.age_at_z(redshift_list, *BPL_COSMO)

    t0 = bpl_t_table[-1]
    lgt0 = np.log10(t0)

    return (
        mah_data["halo_id"],
        mah_data["mpeak_history_main_prog"],
        bpl_t_table,
        BPL_MAH_FIT_LGMIN,
        lgt0,
    )
