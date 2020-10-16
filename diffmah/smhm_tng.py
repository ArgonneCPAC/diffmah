"""Model for the SMHM of TNG central galaxies across time."""
from jax import numpy as jnp
from jax import jit as jjit


@jjit
def _logsm_from_logmhalo_tng(
    logm,
    time,
    logm_crit=11.75,
    ratio=-1.75,
    smhm_k_logm=1.6,
    lo=2.5,
    hi_x0=14.25,
    hi_k=1.5,
    hi_hi=0.5,
):
    """Stellar-to-halo-mass relation for TNG centrals.

    Parameters
    ----------
    logmh : ndarray
        Base-10 log of halo mass in Msun

    time : ndarray
        Age of the Universe in Gyr

    Returns
    -------
    logsm : ndarray
        Base-10 log of stellar mass in Msun

    """
    logsm_at_logm_crit = logm_crit + ratio

    hi = _get_hi_index(logm, time, hi_x0, hi_k, hi_hi)
    numerator = hi - lo
    denominator = 1 + jnp.exp(-smhm_k_logm * (logm - logm_crit))
    powerlaw_index = lo + numerator / denominator
    return logsm_at_logm_crit + powerlaw_index * (logm - logm_crit)


def _get_hi_index(logm, time, hi_x0, hi_k, hi_hi):
    hi_lo = _get_hi_index_lo(time)
    return _sigmoid(logm, hi_x0, hi_k, hi_lo, hi_hi)


def _get_hi_index_lo(time):
    return _sigmoid(time, 3, 0.65, 1.0, 0.8)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))
