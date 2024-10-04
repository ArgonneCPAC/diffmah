"""
"""

import numpy as np
from jax import random as jran

from ... import diffmah_kernels
from .. import diffmahpop_params_monocensat as dpp
from .. import monocens_fixed_tpeak_fithelp


def test_loss_grads():
    ran_key = jran.key(0)
    t_obs = 10.0
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    tarr = np.linspace(0.1, t_obs, 100)

    t_peak_target = t_0
    n_singlebin = 175
    t_peak_singlebin = np.linspace(3, 12, n_singlebin)

    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        mah_params = diffmah_kernels.DEFAULT_MAH_PARAMS._replace(logm0=lgm_obs)
        __, mean_log_mah = diffmah_kernels.mah_singlehalo(
            mah_params, tarr, t_peak_target, lgt0
        )
        std_log_mah = np.zeros_like(mean_log_mah) + 0.5
        target_frac_peaked = np.zeros_like(mean_log_mah) + 0.5
        args = (
            dpp.DEFAULT_DIFFMAHPOP_U_PARAMS,
            tarr,
            lgm_obs,
            t_obs,
            t_peak_singlebin,
            ran_key,
            lgt0,
            mean_log_mah,
            std_log_mah,
            target_frac_peaked,
        )
        loss = monocens_fixed_tpeak_fithelp._loss_mah_moments_singlebin_u_params(*args)
        assert np.all(np.isfinite(loss))
