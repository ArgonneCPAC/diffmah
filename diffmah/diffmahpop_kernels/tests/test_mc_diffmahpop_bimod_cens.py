"""
"""

import numpy as np
from jax import random as jran

from ... import diffmah_kernels as dk
from ...defaults import LGT0
from .. import mc_bimod_cens as mcdpk
from ..bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS

EPS = 1e-4


def test_mc_mean_diffmah_params_are_always_in_bounds():
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    lgmarr = np.linspace(10, 16, 20)
    for lgm_obs in lgmarr:
        _res = mcdpk._mean_diffmah_params(
            DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
        )
        mah_params_e, mah_params_l, fec = _res

        assert np.all(mah_params_e.t_peak > 0)
        assert np.all(mah_params_e.t_peak <= t_0 + EPS)
        assert np.all(mah_params_l.t_peak > 0)
        assert np.all(mah_params_l.t_peak <= t_0 + EPS)

        assert np.all(fec >= 0)
        assert np.all(fec <= 1)

        for p, bound in zip(mah_params_e, dk.MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(mah_params_e, dk.MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(mah_params_l, dk.MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])
        for p, bound in zip(mah_params_l, dk.MAH_PBOUNDS):
            assert np.all(bound[0] < p)
            assert np.all(p < bound[1])


def test_mean_diffmah_params():
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    for lgm_obs in np.linspace(10, 16, 20):
        args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
        _res = mcdpk._mean_diffmah_params(*args)
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
        mah_params_e, mah_params_l, frac_early = _res
        assert np.all(np.isfinite(mah_params_e.logtc))
        assert np.all(np.isfinite(mah_params_l.logtc))


def test_mc_diffmah_params_singlecen_agrees_with_fixed_t_peak_version():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk.mc_diffmah_params_singlecen(*args)
        mah_params_e, mah_params_l, frac_early = _res

        _res2 = mcdpk.mc_diffmah_params_singlecen(
            *args,
            t_peak=mah_params_e.t_peak,
        )
        mah_params_e2, mah_params_l2, frac_early2 = _res2
        for p, p2 in zip(mah_params_e, mah_params_e2):
            assert np.allclose(p, p2)

        _res3 = mcdpk.mc_diffmah_params_singlecen(
            *args,
            t_peak=mah_params_l.t_peak,
        )
        mah_params_e3, mah_params_l3, frac_early3 = _res3
        for p, p2 in zip(mah_params_l, mah_params_l3):
            assert np.allclose(p, p2)


def test_mc_diffmah_cenpop():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)

    n_halos = 450
    lgm_key, t_obs_key, t_peak_key, ran_key = jran.split(ran_key, 4)
    lgm_obs = jran.uniform(lgm_key, minval=10, maxval=15, shape=(n_halos,))
    t_obs = jran.uniform(t_obs_key, minval=2, maxval=15, shape=(n_halos,))
    t_peak = jran.uniform(t_obs_key, minval=2, maxval=15, shape=(n_halos,))

    args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, lgt0
    _res = mcdpk.mc_diffmah_cenpop(*args)
    mah_params, mah_params_early, mah_params_late, frac_early_cens, mc_early = _res
    for x in mah_params:
        assert x.shape == (n_halos,)
        assert np.all(np.isfinite(x))

    halopop = mcdpk.mc_diffmah_cenpop(*args, t_peak=t_peak)
    mah_params, mah_params_early, mah_params_late, frac_early_cens, mc_early = halopop
    assert hasattr(halopop, "mah_params")
    for x in mah_params:
        assert x.shape == (n_halos,)
        assert np.all(np.isfinite(x))


def test_mc_diffmah_cenpop_holds_t_peak_fixed_correctly():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)

    n_halos = 4_500
    lgm_key, t_obs_key, t_peak_key, ran_key = jran.split(ran_key, 4)
    lgm_obs = jran.uniform(lgm_key, minval=10, maxval=15, shape=(n_halos,))
    t_obs = jran.uniform(t_obs_key, minval=2, maxval=15, shape=(n_halos,))
    t_peak = jran.uniform(t_obs_key, minval=0.2, maxval=10, shape=(n_halos,))

    args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, lgt0
    halopop = mcdpk.mc_diffmah_cenpop(*args, t_peak=t_peak)
    mah_params, mah_params_early, mah_params_late, frac_early_cens, mc_early = halopop
    assert np.allclose(t_peak, mah_params.t_peak, atol=0.01)


def test_predict_mah_moments_singlebin():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    lgmarr = np.linspace(10, 15, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, lgt0)
        mean_log_mah, std_log_mah, f_peaked = mcdpk.predict_mah_moments_singlebin(*args)
        assert np.all(np.isfinite(mean_log_mah))
        assert np.all(np.isfinite(std_log_mah))
        assert np.all(np.isfinite(f_peaked))


def test_mc_diffmah_halo_sample():
    ran_key = jran.key(0)
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    t_obs = 10.0
    tarr = np.linspace(0.1, t_obs, 100)
    lgmarr = np.linspace(10, 15.5, 20)
    for lgm_obs in lgmarr:
        args = (DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, lgt0)
        _res = mcdpk._mc_diffmah_halo_sample(*args)
        _res_early = _res[:3]
        _res_late = _res[3:6]
        frac_early = _res[6]

        # Test early sequence
        (mah_params, dmhdt, log_mah) = _res_early
        assert np.all(np.isfinite(mah_params))

        assert np.all(np.isfinite(mah_params.t_peak))
        assert np.all(mah_params.t_peak > 0.0)
        assert np.all(mah_params.t_peak <= t_0)

        assert np.all(np.isfinite(log_mah))
        assert np.all(np.isfinite(dmhdt))

        # Test late sequence
        (mah_params, dmhdt, log_mah) = _res_late
        for x, pname in zip(mah_params, mah_params._fields):
            assert np.all(np.isfinite(mah_params)), (lgm_obs, pname)

        assert np.all(np.isfinite(mah_params.t_peak))
        assert np.all(mah_params.t_peak > 0.0)
        assert np.all(mah_params.t_peak <= t_0)

        assert np.all(np.isfinite(log_mah))
        assert np.all(np.isfinite(dmhdt))

        assert np.all(frac_early > 0)
        assert np.all(frac_early < 1)


def test_mc_cenpop():
    t0 = 10**LGT0

    n_t = 100
    tarr = np.linspace(0.1, t0, n_t)
    n_halos = 20_000
    ran_key = jran.key(0)
    halo_key, lgm_key, t_key = jran.split(ran_key, 3)
    lgm_obs = jran.uniform(lgm_key, minval=10, maxval=15, shape=(n_halos,))
    t_obs = jran.uniform(t_key, minval=2, maxval=t0, shape=(n_halos,))

    _res = mcdpk.mc_cenpop(
        DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, halo_key, LGT0
    )
    for _x in _res:
        assert np.all(np.isfinite(_x))
    mah_params, dmhdt, log_mah = _res
    assert log_mah.shape == (n_halos, n_t)

    dmhdt_recomputed, log_mah_recomputed = dk.mah_halopop(mah_params, tarr, LGT0)
    assert np.allclose(dmhdt, dmhdt_recomputed)
    assert np.allclose(log_mah, log_mah_recomputed)
