"""
"""
import numpy as np
from collections import OrderedDict
from .quenching_times import central_quenching_time
from .quenching_times import DEFAULT_PARAMS as DEFAULT_QTIME_PARAMS
from .quenching_times import QFUNC_PARAMS
from .quenching_times import inverse_central_quenching_time
from .quenching_probability import quenching_prob_cens
from .quenching_probability import DEFAULT_PARAM_VALUES as DEFAULT_QPROB_PARAMS
from .utils import jax_inverse_sigmoid


DEFAULT_QTIME_CENS_PARAMS = OrderedDict(
    [
        (key, DEFAULT_QTIME_PARAMS[key])
        for key in DEFAULT_QTIME_PARAMS.keys()
        if "sat" not in key
    ]
)

DEFAULT_QPROB_CENS_PARAMS = OrderedDict(
    [
        (key, DEFAULT_QPROB_PARAMS[key])
        for key in DEFAULT_QPROB_PARAMS.keys()
        if "sat" not in key
    ]
)


def _get_params(defaults, **kwargs):
    return OrderedDict(
        [(key, kwargs.get(key, defaults.get(key))) for key in defaults.keys()]
    )


def qprob_at_tobs(logm0, tobs, qtime_percentile=0.5, qcut=0.1, **kwargs):
    """Calculate the probability the galaxy is quenched at tobs."""
    assert np.shape(logm0) == (), "logm0 should be a float"
    assert np.shape(tobs) == (), "logm0 should be a float"

    qprob_params = _get_params(DEFAULT_QPROB_CENS_PARAMS, **kwargs)
    qtime_params = _get_params(DEFAULT_QTIME_CENS_PARAMS, **kwargs)

    qtime = central_quenching_time(logm0, qtime_percentile, **qtime_params)
    qprob_z0 = quenching_prob_cens(logm0, **qprob_params)

    qfunc_k, qfunc_ylo, qfunc_yhi = QFUNC_PARAMS.values()
    qs_time = jax_inverse_sigmoid(qcut, qtime, qfunc_k, qfunc_ylo, qfunc_yhi)

    qs_time_percentile = inverse_central_quenching_time(logm0, qs_time, **qtime_params)

    return qs_time_percentile * qprob_z0
