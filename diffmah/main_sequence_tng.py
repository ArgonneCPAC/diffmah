"""Model for the star-forming main sequence of TNG central galaxies across time."""
from jax import numpy as jnp
from jax import jit as jjit


@jjit
def main_sequence_log_ssfr_tng(logsm, time):
    """Mean sSFR across mass and time.

    Parameters
    ----------
    logsm : float or ndarray

    time : float or ndarray

    Returns
    -------
    log_ssfr : ndarray
        Stores base-10 log of SFR/Mstar of star-forming TNG centrals

    """
    x0 = _x0_vs_time(time)
    k = _k_vs_time(time)
    ymin = _ymin_vs_time(time)
    ymax = _ymax_vs_time(time)
    log_ssfr = _sigmoid(logsm, x0, k, ymin, ymax)
    return log_ssfr


@jjit
def main_sequence_scatter_tng(time):
    """Scatter in sSFR across time.

    Parameters
    ----------
    time : float or ndarray

    Returns
    -------
    scatter : ndarray
        Scatter of sSFR in dex of star-forming TNG centrals

    """
    return _sigmoid(time, x0=1.5, k=0.8, ymin=0, ymax=0.3)


@jjit
def _sigmoid(x, x0=0, k=1, ymin=-1, ymax=1):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


def _x0_vs_time(time):
    return _sigmoid(time, x0=8.5, k=0.5, ymin=11, ymax=11.325)


def _k_vs_time(time):
    return _sigmoid(time, x0=8.5, k=0.25, ymin=1.5, ymax=3.4)


def _ymin_vs_time(time):
    return _sigmoid(time, x0=8.5, k=0.1, ymin=-11.1, ymax=-7.1)


def _ymax_vs_time(time):
    return _sigmoid(time, x0=12.0, k=0.1, ymin=-12.5, ymax=-7)
