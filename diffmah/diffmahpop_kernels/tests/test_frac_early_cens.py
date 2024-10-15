"""
"""

import numpy as np
from jax import random as jran

from .. import frac_early_cens as fec

TOL = 1e-3


def test_param_u_param_names_propagate_properly():
    gen = zip(
        fec.DEFAULT_FEC_U_PARAMS._fields,
        fec.DEFAULT_FEC_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = fec.get_bounded_fec_params(fec.DEFAULT_FEC_U_PARAMS)
    assert set(inferred_default_params._fields) == set(fec.DEFAULT_FEC_PARAMS._fields)

    inferred_default_u_params = fec.get_unbounded_fec_params(fec.DEFAULT_FEC_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        fec.DEFAULT_FEC_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        fec.get_bounded_fec_params(fec.DEFAULT_FEC_PARAMS)
        raise NameError("get_bounded_fec_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        fec.get_unbounded_fec_params(fec.DEFAULT_FEC_U_PARAMS)
        raise NameError("get_unbounded_fec_params should not accept u_params")
    except AttributeError:
        pass


def test_default_params_are_in_bounds():
    for key in fec.DEFAULT_FEC_PARAMS._fields:
        val = getattr(fec.DEFAULT_FEC_PARAMS, key)
        bound = getattr(fec.FEC_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_param_u_param_inversion():
    assert np.allclose(
        fec.DEFAULT_FEC_PARAMS,
        fec.get_bounded_fec_params(fec.DEFAULT_FEC_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = fec.get_bounded_fec_params(
        fec.get_unbounded_fec_params(fec.DEFAULT_FEC_PARAMS)
    )
    assert np.allclose(fec.DEFAULT_FEC_PARAMS, inferred_default_params, rtol=TOL)


def test_frac_early_cens_kern():
    ran_key = jran.key(0)
    n_params = len(fec.DEFAULT_FEC_PARAMS)
    n_tests = 100
    for __ in range(n_tests):
        p_key, lgm_key, t_key, ran_key = jran.split(ran_key, 4)
        uran = jran.uniform(p_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = fec.DEFAULT_FEC_U_PARAMS._make(uran)
        params = fec.get_bounded_fec_params(u_params)

        lgm_obs_arr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(1000,))
        t_obs_arr = jran.uniform(t_key, minval=0, maxval=20, shape=(1000,))
        frac_early = fec._frac_early_cens_kern(params, lgm_obs_arr, t_obs_arr)
        assert np.all(np.isfinite(frac_early))
        assert np.all(frac_early >= fec._FBOUNDS[0])
        assert np.all(frac_early <= fec._FBOUNDS[1])
