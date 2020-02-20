"""Model for statistical evolution of halo vmax across time."""
import numpy as np
from numba import njit
from collections import OrderedDict


__all__ = ("vmax_vs_mhalo_and_redshift",)


PARAM_DEFAULT_DICT = OrderedDict(
    a0_vmax=0.378, a0_vmax_alpha=-0.142, a0_vmax_beta=-1.79, logmpiv_vmax=12.21
)

PARAM_BOUNDS_DICT = OrderedDict(
    a0_vmax=(0.05, 0.75),
    a0_vmax_alpha=(-0.5, -0.05),
    a0_vmax_beta=(-4, -1),
    logmpiv_vmax=(11.5, 12.75),
)


def vmax_vs_mhalo_and_redshift(mhalo, redshift, **param_dict):
    """Model for scaling of halo vmax with mass and redshift.

    Relation taken from Equation (E2) from Behroozi+19,
    https://arxiv.org/abs/1806.07893.

    Parameters
    ----------
    mhalo : float or ndarray
        Mass of the halo at the input redshift assuming h=0.7

    redshift : float or ndarray

    vmax : ndarray
        Empty array that will be filled with Vmax [physical km/s]
        for a halo of the input mass and redshift

    """
    mhalo, redshift = _get_1d_arrays(mhalo, redshift)
    vmax = np.zeros_like(mhalo)

    params = OrderedDict()
    for key, default_value in PARAM_DEFAULT_DICT.items():
        params[key] = param_dict.get(key, default_value)

    __ = _vmax_vs_mhalo_and_redshift(mhalo, redshift, vmax, *params.values())
    return vmax


@njit
def _vmax_vs_mhalo_and_redshift(
    mhalo, redshift, vmax, a0_vmax, a0_vmax_alpha, a0_vmax_beta, logmpiv_vmax
):
    n = vmax.size
    for i in range(n):
        m = mhalo[i]
        z = redshift[i]
        a = 1 / (1 + z)

        denom_term1 = (a / a0_vmax) ** a0_vmax_alpha
        denom_term2 = (a / a0_vmax) ** a0_vmax_beta
        mpivot = (10 ** logmpiv_vmax) / (denom_term1 + denom_term2)
        vmax[i] = 200 * (m / mpivot) ** (1 / 3.0)


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length."""
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
