"""
"""
import numpy as np
from collections import OrderedDict
from .quenching_times import central_quenching_time
from .quenching_times import DEFAULT_PARAMS as DEFAULT_QTIME_PARAMS
from .quenching_times import QFUNC_PARAMS
from .quenching_times import qtime_percentile_at_qtime
from .quenching_probability import quenching_prob_cens
from .quenching_probability import DEFAULT_PARAM_VALUES as DEFAULT_QPROB_PARAMS
from .utils import jax_inverse_sigmoid


def _get_params(defaults, **kwargs):
    return OrderedDict(
        [(key, kwargs.get(key, defaults.get(key))) for key in defaults.keys()]
    )


def quenched_fraction_at_tobs(
    logm0, tobs, quenching_percentile=None, qcut=0.1, **kwargs
):
    """Calculate the probability the galaxy is quenched at tobs."""
    assert np.shape(logm0) == (), "logm0 should be a float"
    assert np.shape(tobs) == (), "logm0 should be a float"

    qprob_params = _get_params(DEFAULT_QPROB_PARAMS, **kwargs)
    qtime_params = _get_params(DEFAULT_QTIME_PARAMS, **kwargs)

    qtime = central_quenching_time(logm0, quenching_percentile, **qtime_params)
    qprob_z0 = quenching_prob_cens(logm0, **qprob_params)

    qfunc_k, qfunc_ylo, qfunc_yhi = QFUNC_PARAMS.values()
    qs_time = jax_inverse_sigmoid(qcut, qtime, qfunc_k, qfunc_ylo, qfunc_yhi)

    qtime_cdf = qtime_percentile_at_qtime(logm0, qs_time, **qtime_params)

    return qtime_cdf * qprob_z0
