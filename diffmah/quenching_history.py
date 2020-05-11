"""
"""
from collections import OrderedDict
from jax import numpy as jax_np
from .utils import _get_param_dict

MEDIAN_HISTORY_PARAMS = OrderedDict(
    fms_logtc_x0=12.3,
    fms_logtc_k=1.5,
    fms_logtc_ylo=1.25,
    fms_logtc_yhi=0.475,
    fms_late_x0=13.4,
    fms_late_k=2,
    fms_late_ylo=-0.1,
    fms_late_yhi=-1.85,
)

DEFAULT_HISTORY_PARAMS = OrderedDict(fms_logtc=0.5, fms_k=5, fms_ylo=0, fms_yhi=-1.5)


def mean_log_main_sequence_fraction(logm0, logt, **kwargs):
    """Main-sequence probability vs time for central galaxies.

    Default values tuned to match UniverseMachine.

    Parameters
    ----------
    logm0 : float

    logt : ndarray shape (n, )
        Base-10 log of cosmic time in Gyr

    **params : optional
        Accepts float values for all keyword arguments
        appearing in MEDIAN_HISTORY_PARAMS dictionary.

    Returns
    -------
    log_ms_frac : ndarray shape (n, )
        Base-10 log of the probability the central galaxy is
        on the main sequence at each input time.

    """
    mean_q_param_dict = _get_param_dict(MEDIAN_HISTORY_PARAMS, **kwargs)
    mean_q_params = jax_np.array(list(mean_q_param_dict.values())).astype("f4")
    data = logm0, logt
    return _mean_log_main_sequence_fraction(mean_q_params, data)


def _mean_log_main_sequence_fraction(params, data):
    logm0, logt = data
    fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi = params[0:4]
    fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi = params[4:8]

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


def _get_mah_params(logm0, **kwargs):
    mean_mah_params = _get_params(MEDIAN_HISTORY_PARAMS, **kwargs)
    all_mah_params = jax_np.array(list(mean_mah_params.values())).astype("f4")
    fms_x0 = _fms_logtc_vs_logm0(logm0, *all_mah_params[0:4])
    fms_yhi = _fms_yhi_vs_logm0(logm0, *all_mah_params[4:8])
    return fms_x0, fms_yhi


def _fms_logtc_vs_logm0(logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi):
    return _jax_sigmoid(logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi)


def _fms_yhi_vs_logm0(logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi):
    return _jax_sigmoid(logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi)


def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))


def _get_params(defaults, **kwargs):
    return OrderedDict([(key, kwargs.get(key, val)) for key, val in defaults.items()])
