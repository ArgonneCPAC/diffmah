"""Model for the quenched of TNG central galaxies across time."""
from jax import numpy as jnp
from jax import jit as jjit


@jjit
def fq_tng(logsm, time):
    """Mean fraction of TNG galaxies with sSFR < -11 across mass and time.

    Parameters
    ----------
    logsm : float or ndarray

    time : float or ndarray

    Returns
    -------
    fq : ndarray
        Stores the quenched fraction of TNG centrals

    """
    x0 = _x0_vs_time(time)
    k = _k_vs_time(time)
    ymin = _ymin_vs_time(time)
    ymax = _ymax_vs_time(time)
    log_ssfr = _sigmoid(logsm, x0, k, ymin, ymax)
    return log_ssfr


@jjit
def _sigmoid(x, x0=0, k=1, ymin=-1, ymax=1):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


def _x0_vs_time(time):
    return _sigmoid(time, 4, 0.5, 11.45, 10.55)


def _k_vs_time(time):
    return _sigmoid(time, 2.2, 0.35, -5, 10.35)


def _ymin_vs_time(time):
    return 0.0


def _ymax_vs_time(time):
    return _sigmoid(time, 3.5, 0.5, -0.1, 0.95)
