"""
"""

import numpy as np
from jax import random as jran

from ... import diffmah_kernels
from .. import diffmahpop_params as dpp
from .. import variance_fithelp


def test_loss_grads():
    ran_key = jran.key(0)
    t_obs = 10.0
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    tarr = np.linspace(0.1, t_obs, 100)
    t_peak = t_0

    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        mah_params = diffmah_kernels.DEFAULT_MAH_PARAMS._replace(logm0=lgm_obs)
        __, mean_log_mah = diffmah_kernels.mah_singlehalo(
            mah_params, tarr, t_peak, lgt0
        )
        std_log_mah = np.zeros_like(mean_log_mah) + 0.5
        args = (
            dpp.DEFAULT_DIFFMAHPOP_PARAMS,
            tarr,
            lgm_obs,
            t_obs,
            ran_key,
            lgt0,
            mean_log_mah,
            std_log_mah,
        )
        loss = variance_fithelp._loss_mah_moments_singlebin(*args)
        assert np.all(np.isfinite(loss))
