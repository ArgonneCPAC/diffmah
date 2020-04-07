"""
"""
import numpy as np
from collections import OrderedDict
from .quenching_times import central_quenching_time
from .quenching_times import DEFAULT_CENS_PARAMS as DEFAULT_QTIME_CENS_PARAMS
from .quenching_times import quenching_function
from .quenching_probability import quenching_prob_cens
from .quenching_probability import DEFAULT_CENS_PARAMS as DEFAULT_QPROB_CENS_PARAMS


def _get_params(defaults, **kwargs):
    return OrderedDict(
        [(key, kwargs.get(key, defaults.get(key))) for key in defaults.keys()]
    )


def qprob_at_tobs(logm0, tobs, qtime_percentile=0.5, **kwargs):
    """Calculate the probability the galaxy is quenched at tobs."""
    assert np.shape(logm0) == (), "logm0 should be a float"
    assert np.shape(tobs) == (), "logm0 should be a float"

    qprob_params = _get_params(DEFAULT_QPROB_CENS_PARAMS, **kwargs)
    qtime_params = _get_params(DEFAULT_QTIME_CENS_PARAMS, **kwargs)

    qtime = central_quenching_time(logm0, qtime_percentile, **qtime_params)[0]
    qprob_at_z0 = quenching_prob_cens(logm0, **qprob_params)[0]

    qprob_at_z = qprob_at_z0 * (1 - quenching_function(tobs, qtime))

    return qprob_at_z[0]
