"""
"""
from collections import OrderedDict
from jax import numpy as jax_np

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


def log_ms_fraction_um_median(logm0, lgt, fms_ylo=0, fms_k=7, **kwargs):
    p = _get_params(MEDIAN_HISTORY_PARAMS, **kwargs)

    fms_x0 = _fms_logtc_vs_logm0(
        logm0,
        p["fms_logtc_x0"],
        p["fms_logtc_k"],
        p["fms_logtc_ylo"],
        p["fms_logtc_yhi"],
    )
    fms_yhi = _fms_late_vs_logm0(
        logm0, p["fms_late_x0"], p["fms_late_k"], p["fms_late_ylo"], p["fms_late_yhi"],
    )
    return _jax_sigmoid(lgt, fms_x0, fms_k, fms_ylo, fms_yhi)


def _log_main_sequence_fraction(logt, fms_logtc, fms_late):
    fms_early = 0
    fms_k = 7
    return _jax_sigmoid(logt, fms_logtc, fms_k, fms_early, fms_late)


def _fms_logtc_vs_logm0(logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi):
    return _jax_sigmoid(logm0, fms_logtc_x0, fms_logtc_k, fms_logtc_ylo, fms_logtc_yhi)


def _fms_late_vs_logm0(logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi):
    return _jax_sigmoid(logm0, fms_late_x0, fms_late_k, fms_late_ylo, fms_late_yhi)


def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))


def _get_params(defaults, **kwargs):
    return OrderedDict([(key, kwargs.get(key, val)) for key, val in defaults.items()])
