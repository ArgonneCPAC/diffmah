"""
"""

import warnings

import numpy as np
from jax import random as jran

from .. import diffmah_kernels as dk
from .. import fitting_helpers as fithelp
from ..bfgs_wrapper import bfgs_adam_fallback


def test_fitting_helpers_component_functions():
    t_sim = np.linspace(0.1, 13.8, 2000)
    logt = np.log10(t_sim)
    logt0 = logt[-1]
    lgm_min = 7.0
    logm0 = 13.0
    logtc = 0.4
    early, late = 0.8, 0.15
    t_peak = 13.0
    p_true = dk.DiffmahParams(logm0, logtc, early, late, t_peak)
    mah_sim = 10 ** dk.mah_singlehalo(p_true, t_sim, logt0)[1]

    u_p_init, loss_data, skip_fit = fithelp.get_loss_data(t_sim, mah_sim, lgm_min)
    t_target, log_mah_target, u_t_peak, logt0_out = loss_data
    t_peak_inferred = dk._get_bounded_diffmah_param(u_t_peak, dk.MAH_PBOUNDS.t_peak)
    assert np.all(np.isfinite(u_p_init))
    assert np.allclose(logt0, logt0_out)
    assert np.allclose(t_peak_inferred, p_true.t_peak, atol=0.05)

    msk = t_sim >= t_target.min()
    assert np.allclose(log_mah_target, np.log10(mah_sim[msk]))

    _res = bfgs_adam_fallback(fithelp.loss_and_grads_kern, u_p_init, loss_data)
    u_p_best, loss_best, fit_terminates, code_used = _res

    ATOL = 0.1

    assert not skip_fit
    assert fit_terminates
    assert code_used == 0
    assert loss_best < 0.001
    p_best_inferred = dk.get_bounded_mah_params(dk.DiffmahUParams(*u_p_best, u_t_peak))
    assert np.allclose(p_best_inferred, p_true, rtol=ATOL)


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
        mah_sim = 10 ** dk.mah_singlehalo(mah_params, t_sim, LGT0)[1]

        log_mah_noise = jran.uniform(
            noise_key, minval=-0.1, maxval=0.1, shape=mah_sim.shape
        )
        mah_target = 10 ** (np.log10(mah_sim) + log_mah_noise)

        _res = fithelp.diffmah_fitter(t_sim, mah_target, lgm_min)
        p_best, loss_best, skip_fit, fit_terminates, code_used, loss_data = _res
        __, log_mah_fit = dk.mah_singlehalo(p_best, t_sim, LGT0)
        loss_check = fithelp._mse(log_mah_fit, np.log10(mah_sim))
        assert loss_check < 0.01


def test_diffmah_fitter_skips_mahs_with_insufficient_data():
    t_sim = np.linspace(0.1, 13.8, 100)
    mah_sim = 10 ** np.linspace(0, 14, t_sim.size)
    mah_sim[:10] = 0.0
    lgm_min = 100
    _res = fithelp.diffmah_fitter(t_sim, mah_sim, lgm_min=lgm_min)
    p_best, loss_best, skip_fit, fit_terminates, code_used, loss_data = _res
    assert skip_fit
    assert fit_terminates is False
    assert code_used == -1
    assert loss_best == fithelp.NOFIT_FILL
    assert np.allclose(p_best, fithelp.NOFIT_FILL)


def test_get_target_data():
    t_sim = np.linspace(0.1, 13.8, 100)
    log_mah_sim = np.linspace(1, 14, t_sim.size)
    dlogm_cut = 20.0
    t_fit_min = 0.0

    # No data points should be excluded
    lgm_min = 0.0
    logt_target, log_mah_target = fithelp._get_target_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    assert log_mah_target.size == log_mah_sim.size

    # No data points should be excluded
    lgm_min = -float("inf")
    logt_target, log_mah_target = fithelp._get_target_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    assert log_mah_target.size == log_mah_sim.size

    # First data point should be excluded
    lgm_min = 1.001
    logt_target, log_mah_target = fithelp._get_target_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    assert log_mah_target.size == log_mah_sim.size - 1

    # All data points should be excluded
    lgm_min = 21.001
    logt_target, log_mah_target = fithelp._get_target_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    assert len(log_mah_target) == 0


def test_get_loss_data():
    t_sim = np.linspace(0.1, 13.8, 100)
    mah_sim = 10 ** np.linspace(1, 14, t_sim.size)

    # No data points should be excluded
    u_p_init, loss_data, skip_fit = fithelp.get_loss_data(
        t_sim, mah_sim, dlogm_cut=100.0, t_fit_min=0.0
    )
    t_target, log_mah_target, u_t_peak, logt0 = loss_data
    assert skip_fit is False
    assert t_target.size == t_sim.size
    assert log_mah_target.size == mah_sim.size
    assert np.all(np.isfinite(u_t_peak))
    t0 = 10**logt0
    assert 5 < t0 < 25
    assert np.all(np.isfinite(u_p_init))
    assert len(u_p_init) == len(dk.DEFAULT_MAH_PARAMS) - 1

    # High-redshift data points should be excluded
    t_fit_min = 2.0
    u_p_init, loss_data, skip_fit = fithelp.get_loss_data(
        t_sim, mah_sim, dlogm_cut=100.0, t_fit_min=t_fit_min
    )
    assert skip_fit is False
    t_target = loss_data[0]
    msk = t_sim > t_fit_min
    assert np.allclose(t_target, t_sim[msk])

    # Low-mass data points should be excluded
    dlogm_cut = 3.0
    u_p_init, loss_data, skip_fit = fithelp.get_loss_data(
        t_sim, mah_sim, dlogm_cut=dlogm_cut, t_fit_min=t_fit_min
    )
    assert skip_fit is False
    log_mah_target = loss_data[1]
    msk = mah_sim > 10 ** (np.log10(mah_sim[-1]) - dlogm_cut)
    assert np.allclose(log_mah_target, np.log10(mah_sim[msk]))

    # Fit should be skipped on account of npts_min cut
    npts_min = fithelp.NPTS_FIT_MIN - 2
    lgm_min = np.log10(mah_sim[-npts_min])
    u_p_init, loss_data, skip_fit = fithelp.get_loss_data(
        t_sim, mah_sim, lgm_min=lgm_min, npts_min=npts_min
    )
    assert skip_fit is True

    # Fit should NOT be skipped on account of npts_min cut
    npts_min = fithelp.NPTS_FIT_MIN - 3
    lgm_min = np.log10(mah_sim[-npts_min])
    u_p_init, loss_data, skip_fit = fithelp.get_loss_data(
        t_sim, mah_sim, lgm_min=lgm_min, npts_min=npts_min
    )
    assert skip_fit is False


def test_diffmah_fitter_raises_warning_when_passed_log_mah_instead_of_mah():
    t_sim = np.linspace(0.1, 13.8, 100)
    log_mah_sim = np.linspace(5, 15, t_sim.size)
    with warnings.catch_warnings(record=True) as w:
        fithelp.diffmah_fitter(t_sim, log_mah_sim)
    substr = "Values of input MAH are suspiciously small"
    assert substr in str(w[-1].message)


def test_get_outline_good_fits():
    ran_key = jran.key(0)
    LGT0 = np.log10(13.79)

    t_sim = np.linspace(0.5, 13.8, 100)
    lgm_min = 7.0

    n_tests = 20
    for __ in range(n_tests):
        test_key, noise_key, ran_key = jran.split(ran_key, 3)

        u_p = np.array(dk.DEFAULT_MAH_U_PARAMS)
        uran = jran.uniform(test_key, minval=-20, maxval=20, shape=u_p.shape)
        u_p = dk.DEFAULT_MAH_U_PARAMS._make(u_p + uran)
        mah_params = dk.get_bounded_mah_params(u_p)
        mah_sim = 10 ** dk.mah_singlehalo(mah_params, t_sim, LGT0)[1]

        log_mah_noise = jran.uniform(
            noise_key, minval=-0.5, maxval=0.5, shape=mah_sim.shape
        )
        mah_target = 10 ** (np.log10(mah_sim) + log_mah_noise)

        fit_results = fithelp.diffmah_fitter(t_sim, mah_target, lgm_min)

        outline = fithelp.get_outline(fit_results)
        assert outline[-1] == "\n"
        outdata = fithelp._parse_outline(outline)
        header_colnames = fithelp.HEADER[1:].strip().split()
        assert len(outdata) == len(header_colnames)

        assert len(fithelp.DiffmahFitData._fields) == len(outdata)

        ATOL = 1e-4
        assert np.allclose(outdata.logm0, fit_results.p_best.logm0, rtol=ATOL)
        assert np.allclose(outdata.logtc, fit_results.p_best.logtc, rtol=ATOL)
        assert np.allclose(
            outdata.early_index, fit_results.p_best.early_index, rtol=ATOL
        )
        assert np.allclose(outdata.late_index, fit_results.p_best.late_index, rtol=ATOL)
        assert np.allclose(outdata.t_peak, fit_results.p_best.t_peak, rtol=ATOL)


def test_get_outline_bad_fit():
    """Set up and check a hard-coded example of a no-fit MAH"""
    t_sim = np.array((2.0, 5.0, 13.0))
    mah_sim = 10 ** np.array([14.5, 14.75, 15.0])
    fit_results = fithelp.diffmah_fitter(t_sim, mah_sim)
    outline = fithelp.get_outline(fit_results)
    assert outline[-1] == "\n"
    outdata = fithelp._parse_outline(outline)
    for pname in dk.DEFAULT_MAH_PARAMS._fields:
        val = getattr(outdata, pname)
        assert val == fithelp.NOFIT_FILL
    assert outdata.loss == fithelp.NOFIT_FILL

    assert outdata.n_points_per_fit == 2, fit_results.loss_data.log_mah_target
    assert outdata.fit_algo == -1
