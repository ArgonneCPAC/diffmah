"""
"""

import numpy as np
import pytest
from jax import random as jran

from ...diffmah_kernels import MAH_PBOUNDS
from .. import mc_diffmahpop_kernels_censat as mcdpk
from ..diffmahpop_params_censat import DEFAULT_DIFFMAHPOP_PARAMS


def test_mc_mean_diffmah_params_are_always_in_bounds():
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        _res = mcdpk.mc_mean_diffmah_params(
            DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
        )
        (
            dmah_tpt0_cens,
            dmah_tp_cens,
            t_peak_cens,
            frac_tpt0_cens,
            mc_tpt0_cens,
            t_peak_sats,
            dmah_sats,
        ) = _res
        assert np.all(t_peak_cens > 0)
        assert np.all(t_peak_cens <= t_0)
        assert np.any(t_peak_cens < t_0)

        assert np.all(t_peak_sats > 0)
        assert np.all(t_peak_sats < t_obs)

        assert frac_tpt0_cens.shape == ()
        assert np.all(frac_tpt0_cens >= 0)
        assert np.all(frac_tpt0_cens <= 1)
        assert np.any(frac_tpt0_cens > 0)
        assert np.any(frac_tpt0_cens < 1)

        for p, bound in zip(dmah_tpt0_cens, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(dmah_tp_cens, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(dmah_sats, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])


def test_mc_mean_diffmah_params():
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    for lgm_obs in np.linspace(10, 16, 20):
        args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
        _res = mcdpk.mc_mean_diffmah_params(*args)
        for _x in _res:
            assert np.all(np.isfinite(_x))


def test_mc_diffmah_params_single_censat():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk.mc_diffmah_params_single_censat(*args)
        (
            mah_params_tpt0_cens,
            mah_params_tp_cens,
            t_peak_cens,
            ftpt0_cens,
            mc_tpt0_cens,
            t_peak_sats,
            mah_params_sats,
        ) = _res
        assert np.all(np.isfinite(mah_params_tpt0_cens.logtc))
        assert np.all(np.isfinite(mah_params_tp_cens.logtc))
        assert np.all(np.isfinite(mah_params_sats.logtc))


def test_predict_mah_moments_singlebin_censat():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk.predict_mah_moments_singlebin_censat(*args)
        for _x in _res:
            assert np.all(np.isfinite(_x))


def test_mc_diffmah_halo_sample():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk._mc_diffmah_halo_sample_censat(*args)
        (
            mah_params_tpt0_cens,
            mah_params_tp_cens,
            t_peak_cens,
            frac_tpt0_cens,
            mc_tpt0_cens,
            dmhdt_tpt0_cens,
            log_mah_tpt0_cens,
            dmhdt_tp_cens,
            log_mah_tp_cens,
            dmhdt_sats,
            log_mah_sats,
        ) = _res
        assert np.all(np.isfinite(mah_params_tpt0_cens))
        assert np.all(np.isfinite(mah_params_tp_cens))

        assert np.all(np.isfinite(t_peak_cens))
        assert np.all(t_peak_cens > 0.0)
        assert np.all(t_peak_cens <= t_0)

        assert np.all(np.isfinite(frac_tpt0_cens))
        assert np.all(frac_tpt0_cens > 0.0)
        assert np.all(frac_tpt0_cens < 1.0)

        assert np.all(np.isfinite(log_mah_tpt0_cens))
        assert np.all(np.isfinite(log_mah_tp_cens))
