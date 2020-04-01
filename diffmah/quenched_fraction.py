"""
"""
import numpy as np
from .in_situ_history import in_situ_galaxy_halo_history
from .quenching_times import central_quenching_time
from .quenching_probability import quenching_prob_cens
from .quenching_probability import DEFAULT_PARAM_VALUES as DEFAULT_QPROB_PARAMS


def quenched_fraction_at_tobs(logm0, tobs, quenching_percentile=None, **kwargs):
    """Calculate the probability the galaxy is quenched at tobs."""
    assert np.shape(logm0) == (), "logm0 should be a float"
    assert np.shape(tobs) == (), "logm0 should be a float"

    _X = in_situ_galaxy_halo_history(logm0, **kwargs)
    zarr, tarr, mah, dmhdt = _X[:4]
    sfrh_ms, sfrh_q, smh_ms, smh_q = _X[4:]

    qtime = central_quenching_time(logm0, quenching_percentile, **kwargs)

    default_params_qprob_cens = {
        key: DEFAULT_QPROB_PARAMS[key]
        for key in DEFAULT_QPROB_PARAMS.keys()
        if "cens" in key
    }
    qprob_cens_param_dict = {
        key: kwargs.get(key, default_params_qprob_cens.get(key))
        for key in default_params_qprob_cens.keys()
    }
    qprob_z0 = quenching_prob_cens(logm0, **qprob_cens_param_dict)

    return qfrac
