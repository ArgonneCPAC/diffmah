"""
"""

import numpy as np
from jax import random as jran

from .. import diffmah_kernels as dk
from .. import fitting_helpers as fithelp
from ..bfgs_wrapper import bfgs_adam_fallback


def test_fitting_helpers_integration():
    t_sim = np.linspace(0.1, 13.8, 2000)
    logt = np.log10(t_sim)
    logt0 = logt[-1]
    lgm_min = 7.0
    logm0 = 13.0
    logtc = 0.4
    early, late = 0.8, 0.15
    t_peak = 13.0
    p_true = dk.DiffmahParams(logm0, logtc, early, late, t_peak)
    log_mah_sim = dk.mah_singlehalo(p_true, t_sim, logt0)[1]

    u_p_init, loss_data = fithelp.get_loss_data(t_sim, log_mah_sim, lgm_min)
    t_target, log_mah_target, u_t_peak, logt0_out = loss_data
    t_peak_inferred = dk._get_bounded_diffmah_param(u_t_peak, dk.MAH_PBOUNDS.t_peak)
    assert np.all(np.isfinite(u_p_init))
    assert np.allclose(logt0, logt0_out)
    assert np.allclose(t_peak_inferred, p_true.t_peak, atol=0.05)

    msk = t_sim >= t_target.min()
    assert np.allclose(log_mah_target, log_mah_sim[msk])

    _res = bfgs_adam_fallback(fithelp.loss_and_grads_kern, u_p_init, loss_data)
    u_p_best, loss_best, fit_terminates, code_used = _res

    ATOL = 0.1

    assert fit_terminates
    assert code_used == 0
    assert loss_best < 0.001
    p_best_inferred = dk.get_bounded_mah_params(dk.DiffmahUParams(*u_p_best, u_t_peak))
    assert np.allclose(p_best_inferred, p_true, rtol=ATOL)
    npts_mah = log_mah_target.size

    root_indx = 123
    outline = fithelp.get_outline(
        root_indx, loss_data, u_p_best, loss_best, npts_mah, code_used
    )
    outdata = [float(x) for x in outline.strip().split()]
    header_data = fithelp.HEADER[1:].strip().split()
    assert len(header_data) == len(outdata)
    assert np.allclose(outdata[1], logm0, rtol=ATOL)
    assert np.allclose(outdata[2], logtc, rtol=ATOL)
    assert np.allclose(outdata[3], early, rtol=ATOL)
    assert np.allclose(outdata[4], late, rtol=ATOL)
    assert np.allclose(outdata[5], t_peak, rtol=ATOL)


def test_diffmah_fitter():
    ran_key = jran.key(0)
    LGT0 = np.log10(13.79)

    t_sim = np.linspace(0.5, 13.8, 100)
    lgm_min = 7.0

    n_tests = 10
    for __ in range(n_tests):
        test_key, noise_key, ran_key = jran.split(ran_key, 3)

        u_p = np.array(dk.DEFAULT_MAH_U_PARAMS)
        uran = jran.uniform(test_key, minval=-10, maxval=10, shape=u_p.shape)
        u_p = dk.DEFAULT_MAH_U_PARAMS._make(u_p + uran)
        mah_params = dk.get_bounded_mah_params(u_p)
        __, log_mah_sim = dk.mah_singlehalo(mah_params, t_sim, LGT0)

        log_mah_noise = jran.uniform(
            noise_key, minval=-0.1, maxval=0.1, shape=log_mah_sim.shape
        )
        log_mah_target = log_mah_sim + log_mah_noise

        _res = fithelp.diffmah_fitter(t_sim, log_mah_target, lgm_min)
        p_best, loss_best, fit_terminates, code_used = _res
        __, log_mah_fit = dk.mah_singlehalo(p_best, t_sim, LGT0)
        loss_check = fithelp._mse(log_mah_fit, log_mah_sim)
        assert loss_check < 0.01
