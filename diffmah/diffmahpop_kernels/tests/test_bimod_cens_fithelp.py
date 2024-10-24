"""
"""

import numpy as np
from jax import random as jran

from ... import diffmah_kernels
from .. import bimod_cens_fithelp
from .. import bimod_censat_params as dpp


def test_loss_grads():
    ran_key = jran.key(0)
    t_obs = 10.0
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    tarr = np.linspace(0.1, t_obs, 100)

    lgmarr = np.linspace(10, 15.5, 20)
    for lgm_obs in lgmarr:
        mah_params = diffmah_kernels.DEFAULT_MAH_PARAMS._replace(logm0=lgm_obs)
        __, mean_log_mah = diffmah_kernels.mah_singlehalo(mah_params, tarr, lgt0)
        std_log_mah = np.zeros_like(mean_log_mah) + 0.1
        target_frac_peaked = np.zeros_like(mean_log_mah) + 0.5
        args = (
            dpp.DEFAULT_DIFFMAHPOP_PARAMS,
            tarr,
            lgm_obs,
            t_obs,
            ran_key,
            lgt0,
            mean_log_mah,
            std_log_mah,
            target_frac_peaked,
        )
        loss = bimod_cens_fithelp._loss_mah_moments_singlebin(*args)
        assert np.all(np.isfinite(loss))
