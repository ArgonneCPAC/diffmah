"""
"""

import numpy as np
from jax import random as jran

from ...diffmah_kernels import MAH_PBOUNDS
from .. import mc_diffmahpop_kernels_monocens_fixed_tpeak as mcdpk
from ..diffmahpop_params_monocensat import DEFAULT_DIFFMAHPOP_PARAMS

EPS = 1e-4


def test_mc_mean_diffmah_params_are_always_in_bounds():
    t_obs = 10.0
    lgmarr = np.linspace(10, 16, 20)
    t_peak = 10.0
    for lgm_obs in lgmarr:
        mah_params = mcdpk.mc_mean_diffmah_params(
            DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, t_peak
        )
        for p, bound in zip(mah_params, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(mah_params, MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])


def test_mc_mean_diffmah_params():
    t_obs = 10.0
    t_peak = 8.0
    for lgm_obs in np.linspace(10, 16, 20):
        args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, t_peak
        _res = mcdpk.mc_mean_diffmah_params(*args)
        for _x in _res:
            assert np.all(np.isfinite(_x))


def test_mc_diffmah_params_singlecen():
    ran_key = jran.key(0)
    t_obs = 10.0
    t_peak = 8.0
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, t_peak, ran_key)
        mah_params = mcdpk.mc_diffmah_params_singlecen(*args)
        assert np.all(np.isfinite(mah_params.logtc))


def test_predict_mah_moments_singlebin():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    n_sample = 20
    lgmarr = np.linspace(10, 15, n_sample)
    t_peak_sample = np.linspace(3, 10, n_sample)
    for lgm_obs in lgmarr:
        args = (
            DEFAULT_DIFFMAHPOP_PARAMS,
            tarr,
            lgm_obs,
            t_obs,
            t_peak_sample,
            ran_key,
            lgt0,
        )
        mean_log_mah, std_log_mah, f_peaked = mcdpk.predict_mah_moments_singlebin(*args)
        assert np.all(np.isfinite(mean_log_mah))
        assert np.all(np.isfinite(std_log_mah))
        assert np.all(np.isfinite(f_peaked))


def test_mc_diffmah_halo_sample():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    n_t = 100
    tarr = np.linspace(0.1, t_obs, n_t)
    n_test = 20
    lgmarr = np.linspace(10, 15, n_test)
    n_sample = 50
    t_peak_sample = np.linspace(3, 10, n_sample)
    for lgm_obs in lgmarr:
        args = (
            DEFAULT_DIFFMAHPOP_PARAMS,
            tarr,
            lgm_obs,
            t_obs,
            t_peak_sample,
            ran_key,
            lgt0,
        )
        _res = mcdpk._mc_diffmah_halo_sample(*args)
        (mah_params, dmhdt, log_mah) = _res
        assert log_mah.shape == (n_sample, n_t)
        assert np.all(np.isfinite(mah_params))

        assert np.all(np.isfinite(log_mah))
        assert np.all(np.isfinite(dmhdt))
