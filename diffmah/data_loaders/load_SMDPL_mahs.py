"""Module loads mass assembly history data for diffmah
"""

import numpy as np
import os

BEBOP_SMDPL_nomerging = os.path.join(
    "/lcrc/project/halotools/SMDPL/",
    "dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/",
)

BEBOP_SMDPL_DR1 = os.path.join(
    "/lcrc/project/halotools/UniverseMachine/SMDPL/",
    "sfh_binaries_dr1_bestfit/a_1.000000/",
)

try:
    from umachine_pyio.load_mock import load_mock_from_binaries

    HAS_UM_LOADER = True
except ImportError:
    HAS_UM_LOADER = False


def load_SMDPL_data(subvols, data_drn=BEBOP_SMDPL_DR1):
    """Load the stellar mass histories from UniverseMachine simulation
    applied to the SMDPL simulation.

    Parameters
    ----------
    subvol : nd.array
        Subvolume number in SMDPL
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    SMDPL_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr
    mahs: ndarray of shape (n_gal, n_times)
        Mass accretion histories in Msun/yr .
    """
    if not HAS_UM_LOADER:
        raise ImportError("Must have umachine_pyio installed to load this dataset")

    galprops = ["halo_id", "mpeak_history_main_prog"]
    subvols = np.atleast_1d(subvols)
    mock = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)

    SMDPL_t = np.loadtxt(os.path.join(data_drn, "smdpl_cosmic_time.txt"))

    lgmh_min = 7.0
    mh_min = 10**lgmh_min
    msk = mock["mpeak_history_main_prog"] < mh_min
    clipped_mahs = np.where(msk, 1.0, mock["mpeak_history_main_prog"])
    mahs = np.maximum.accumulate(clipped_mahs, axis=1)

    # logmp = log_mahs[:, -1]
    assert mahs.max() > 1e5

    return SMDPL_t, mahs
