"""
"""

import numpy as np
from jax import random as jran

from ...diffmah_kernels import MAH_PBOUNDS
from .. import mc_diffmahpop_kernels as mcdpk
from ..diffmahpop_params import DEFAULT_DIFFMAHPOP_PARAMS


def test_mc_mean_diffmah_params_are_always_in_bounds():
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0 = mcdpk.mc_mean_diffmah_params(
            DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
        )
        assert np.all(t_peak > 0)
        assert np.all(t_peak <= t_0)
        assert np.any(t_peak < t_0)

        assert ftpt0.shape == ()
        assert np.all(ftpt0 >= 0)
        assert np.all(ftpt0 <= 1)
        assert np.any(ftpt0 > 0)
        assert np.any(ftpt0 < 1)

        for p, bound in zip(dmah_tpt0, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(dmah_tp, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])


def test_mc_mean_diffmah_params():
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    for lgm_obs in np.linspace(10, 16, 20):
        args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
        _res = mcdpk.mc_mean_diffmah_params(*args)
        ran_diffmah_params_tpt0, ran_diffmah_params_tp, t_peak, ftpt0, mc_tpt0 = _res
        for _x in _res:
            assert np.all(np.isfinite(_x))


def test_mc_diffmah_params_singlecen():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk.mc_diffmah_params_singlecen(*args)
        mah_params_tpt0, mah_params_tp, t_peak, ftpt0, mc_tpt0 = _res
        assert np.all(np.isfinite(mah_params_tpt0.logtc))
        assert np.all(np.isfinite(mah_params_tp.logtc))


def test_predict_mah_moments_singlebin():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, lgt0)
        mean_log_mah, std_log_mah = mcdpk.predict_mah_moments_singlebin(*args)
        assert np.all(np.isfinite(mean_log_mah))
        assert np.all(np.isfinite(std_log_mah))


def test_mc_diffmah_halo_sample():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk._mc_diffmah_halo_sample(*args)
        (
            mah_params_tpt0,
            mah_params_tp,
            t_peak,
            ftpt0,
            mc_tpt0,
            dmhdt_tpt0,
            log_mah_tpt0,
            dmhdt_tp,
            log_mah_tp,
        ) = _res
        assert np.all(np.isfinite(mah_params_tpt0))
        assert np.all(np.isfinite(mah_params_tp))

        assert np.all(np.isfinite(t_peak))
        assert np.all(t_peak > 0.0)
        assert np.all(t_peak <= t_0)

        assert np.all(np.isfinite(ftpt0))
        assert np.all(ftpt0 > 0.0)
        assert np.all(ftpt0 < 1.0)

        assert np.all(np.isfinite(log_mah_tpt0))
        assert np.all(np.isfinite(log_mah_tp))
