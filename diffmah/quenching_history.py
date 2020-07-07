"""Module implementing the mean_log_main_sequence_fraction function.
"""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np

MEAN_Q_PARAMS = OrderedDict(
    fms_logtc_x0=11.85,
    fms_logtc_k=0.9,
    fms_logtc_ylo=0.55,
    fms_logtc_yhi=0.49,
    fms_late_x0=13.78,
    fms_late_k=2.25,
    fms_late_ylo=0.11,
    fms_late_yhi=-3.87,
)

DEFAULT_Q_PARAMS = OrderedDict(fms_logtc=0.5, fms_k=5.0, fms_ylo=0.0, fms_yhi=-1.5)


def mean_log_main_sequence_fraction(
    logt,
    logm0,
    fms_logtc_x0=MEAN_Q_PARAMS["fms_logtc_x0"],
    fms_logtc_k=MEAN_Q_PARAMS["fms_logtc_k"],
    fms_logtc_ylo=MEAN_Q_PARAMS["fms_logtc_ylo"],
    fms_logtc_yhi=MEAN_Q_PARAMS["fms_logtc_yhi"],
    fms_late_x0=MEAN_Q_PARAMS["fms_late_x0"],
    fms_late_k=MEAN_Q_PARAMS["fms_late_k"],
    fms_late_ylo=MEAN_Q_PARAMS["fms_late_ylo"],
    fms_late_yhi=MEAN_Q_PARAMS["fms_late_yhi"],
):
    """Main-sequence probability vs time for central galaxies.

    Default values tuned to match UniverseMachine.

    Parameters
    ----------
    logt : ndarray shape (n, )
        Base-10 log of cosmic time in Gyr

    logm0 : float

    **params : optional
        Accepts float values for all keyword arguments
        appearing in MEAN_Q_PARAMS dictionary.

    Returns
    -------
    log_ms_frac : ndarray shape (n, )
        Base-10 log of the probability the central galaxy is
        on the main sequence at each input time.

    """
    logm0 = float(logm0)
    logt = jax_np.atleast_1d(logt).astype("f4")
    params = jax_np.array(
        (
            fms_logtc_x0,
            fms_logtc_k,
            fms_logtc_ylo,
            fms_logtc_yhi,
            fms_late_x0,
            fms_late_k,
            fms_late_ylo,
            fms_late_yhi,
        )
    ).astype("f4")
    return np.array(_mean_log_main_sequence_fraction(logt, logm0, *params,))


def _mean_log_main_sequence_fraction(
    logt,
    logm0,
    fms_logtc_x0,
    fms_logtc_k,
    fms_logtc_ylo,
    fms_logtc_yhi,
    fms_late_x0,
    fms_late_k,
    fms_late_ylo,
    fms_late_yhi,
):

    fms_x0 = _fms_logtc_vs_logm0(
        logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi
    )
    fms_yhi = _fms_yhi_vs_logm0(
        logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi
    )

    fms_yhi = jax_np.where(fms_yhi > 0, 0, fms_yhi)

    return _jax_sigmoid(logt, fms_x0, 7, 0, fms_yhi)


def _log_main_sequence_fraction(fms_x0, fms_yhi, logt):
    fms_yhi = jax_np.where(fms_yhi > 0, 0, fms_yhi)
    return _jax_sigmoid(logt, fms_x0, 7, 0, fms_yhi)


def _fms_logtc_vs_logm0(logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi):
    return _jax_sigmoid(logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi)


def _fms_yhi_vs_logm0(logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi):
    return _jax_sigmoid(logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi)


def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))
