"""
"""

import numpy as np
from jax import random as jran

from .. import logm0_pop_bimod_sats as m0pop

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(
        m0pop.DEFAULT_LOGM0POP_U_PARAMS._fields,
        m0pop.DEFAULT_LOGM0POP_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = m0pop.get_bounded_m0pop_params(
        m0pop.DEFAULT_LOGM0POP_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        m0pop.DEFAULT_LOGM0POP_PARAMS._fields
    )

    inferred_default_u_params = m0pop.get_unbounded_m0pop_params(
        m0pop.DEFAULT_LOGM0POP_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        m0pop.DEFAULT_LOGM0POP_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        m0pop.get_bounded_m0pop_params(m0pop.DEFAULT_LGM0POP_PARAMS)
        raise NameError("get_bounded_m0pop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        m0pop.get_unbounded_m0pop_params(m0pop.DEFAULT_LOGM0POP_U_PARAMS)
        raise NameError("get_unbounded_m0pop_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        m0pop.DEFAULT_LOGM0POP_PARAMS,
        m0pop.get_bounded_m0pop_params(m0pop.DEFAULT_LOGM0POP_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = m0pop.get_bounded_m0pop_params(
        m0pop.get_unbounded_m0pop_params(m0pop.DEFAULT_LOGM0POP_PARAMS)
    )
    assert np.allclose(m0pop.DEFAULT_LOGM0POP_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in m0pop.DEFAULT_LOGM0POP_PARAMS._fields:
        val = getattr(m0pop.DEFAULT_LOGM0POP_PARAMS, key)
        bound = getattr(m0pop.LGM0POP_BOUNDS, key)
        assert bound[0] < val < bound[1]


def test_pred_logm0_kern():
    ran_key = jran.key(0)
    n_tests = 1_000
    for __ in range(n_tests):
        lgm_key, t_obs_key, t_peak_key, ran_key = jran.split(ran_key, 4)
        lgm_obs = jran.uniform(lgm_key, minval=5, maxval=16, shape=())
        t_obs = jran.uniform(t_obs_key, minval=1, maxval=20, shape=())
        t_peak = jran.uniform(t_peak_key, minval=1.5, maxval=20, shape=())
        lgm0_early = m0pop._pred_logm0_kern_early(
            m0pop.DEFAULT_LOGM0POP_PARAMS, lgm_obs, t_obs, t_peak
        )
        assert lgm0_early.shape == ()
        assert np.isfinite(lgm0_early)
        assert lgm0_early > 0
        assert lgm0_early < 20

        lgm0_late = m0pop._pred_logm0_kern_late(
            m0pop.DEFAULT_LOGM0POP_PARAMS, lgm_obs, t_obs, t_peak
        )
        assert lgm0_late.shape == ()
        assert np.isfinite(lgm0_late)
        assert lgm0_late > 0
        assert lgm0_late < 20
