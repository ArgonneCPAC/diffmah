"""
"""

import numpy as np
from jax import random as jran

from .. import tp_pdf_cens_flex as tpc

TOL = 1e-3


def test_param_u_param_names_propagate_properly():
    gen = zip(
        tpc.DEFAULT_TPCENS_U_PARAMS._fields,
        tpc.DEFAULT_TPCENS_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = tpc.get_bounded_tp_cens_params(
        tpc.DEFAULT_TPCENS_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        tpc.DEFAULT_TPCENS_PARAMS._fields
    )

    inferred_default_u_params = tpc.get_unbounded_tp_cens_params(
        tpc.DEFAULT_TPCENS_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        tpc.DEFAULT_TPCENS_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        tpc.get_bounded_tp_pdf_params(tpc.DEFAULT_TPCENS_PARAMS)
        raise NameError("get_bounded_tp_pdf_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        tpc.get_unbounded_tp_pdf_params(tpc.DEFAULT_TPCENS_U_PARAMS)
        raise NameError("get_unbounded_tp_pdf_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        tpc.DEFAULT_TPCENS_PARAMS,
        tpc.get_bounded_tp_cens_params(tpc.DEFAULT_TPCENS_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = tpc.get_bounded_tp_cens_params(
        tpc.get_unbounded_tp_cens_params(tpc.DEFAULT_TPCENS_PARAMS)
    )
    assert np.allclose(tpc.DEFAULT_TPCENS_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in tpc.DEFAULT_TPCENS_PARAMS._fields:
        val = getattr(tpc.DEFAULT_TPCENS_PARAMS, key)
        bound = getattr(tpc.TPCENS_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_mc_tpeak_singlecen():
    t_0 = 13.0
    ran_key = jran.key(0)
    ran_key, m_key, t_key = jran.split(ran_key, 3)
    lgm_obs = jran.uniform(m_key, minval=10.0, maxval=15.0, shape=())
    t_obs = jran.uniform(t_key, minval=1.0, maxval=15.0, shape=())
    args = tpc.DEFAULT_TPCENS_PARAMS, lgm_obs, t_obs, ran_key, t_0
    t_peak_mc_sample = tpc.mc_tpeak_singlecen(*args)
    assert t_peak_mc_sample.shape == ()
    assert np.all(np.isfinite(t_peak_mc_sample))
    assert np.all(t_peak_mc_sample > 0)
    assert np.all(t_peak_mc_sample <= t_0)


def test_mc_t_peak_cenpop():
    t_0 = 13.0
    ran_key = jran.key(0)
    ran_key, m_key, t_key = jran.split(ran_key, 3)
    n_gals = int(1e4)
    lgm_obs = jran.uniform(m_key, minval=10.0, maxval=15.0, shape=(n_gals,))
    t_obs = jran.uniform(t_key, minval=1.0, maxval=15.0, shape=(n_gals,))
    args = tpc.DEFAULT_TPCENS_PARAMS, lgm_obs, t_obs, ran_key, t_0
    t_peak_mc_sample = tpc.mc_t_peak_cenpop(*args)
    assert t_peak_mc_sample.shape == (n_gals,)
    assert np.all(np.isfinite(t_peak_mc_sample))
    assert np.all(t_peak_mc_sample <= t_0)
    assert np.all(t_peak_mc_sample > 0), t_peak_mc_sample.min()
