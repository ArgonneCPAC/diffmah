"""
"""

import numpy as np
from jax import random as jran

from .. import ftpt0_cens

TOL = 1e-3


def test_param_u_param_names_propagate_properly():
    gen = zip(
        ftpt0_cens.DEFAULT_FTPT0_U_PARAMS._fields,
        ftpt0_cens.DEFAULT_FTPT0_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = ftpt0_cens.get_bounded_ftpt0_params(
        ftpt0_cens.DEFAULT_FTPT0_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        ftpt0_cens.DEFAULT_FTPT0_PARAMS._fields
    )

    inferred_default_u_params = ftpt0_cens.get_unbounded_ftpt0_params(
        ftpt0_cens.DEFAULT_FTPT0_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        ftpt0_cens.DEFAULT_FTPT0_U_PARAMS._fields
    )


def test_get_bounded_ftpt0_params_fails_when_passing_params():
    try:
        ftpt0_cens.get_bounded_ftpt0_params(ftpt0_cens.DEFAULT_FTPT0_PARAMS)
        raise NameError("get_bounded_ftpt0_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_ftpt0_params_fails_when_passing_u_params():
    try:
        ftpt0_cens.get_unbounded_tpk_params(ftpt0_cens.DEFAULT_FTPT0_U_PARAMS)
        raise NameError("get_unbounded_tpk_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        ftpt0_cens.DEFAULT_FTPT0_PARAMS,
        ftpt0_cens.get_bounded_ftpt0_params(ftpt0_cens.DEFAULT_FTPT0_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = ftpt0_cens.get_bounded_ftpt0_params(
        ftpt0_cens.get_unbounded_ftpt0_params(ftpt0_cens.DEFAULT_FTPT0_PARAMS)
    )
    assert np.allclose(
        ftpt0_cens.DEFAULT_FTPT0_PARAMS, inferred_default_params, rtol=TOL
    )


def test_default_params_are_in_bounds():
    for key in ftpt0_cens.DEFAULT_FTPT0_PDICT.keys():
        val = getattr(ftpt0_cens.DEFAULT_FTPT0_PARAMS, key)
        bound = getattr(ftpt0_cens.FTPT0_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_ftpt0_kernel_is_bounded_default_params():
    ntests = 100
    ngals = 2_000

    ran_key = jran.key(0)

    for __ in range(ntests):
        ran_key, lgm_key, tobs_key = jran.split(ran_key, 3)
        lgm_obs = jran.uniform(lgm_key, minval=5, maxval=20, shape=(ngals,))
        t_obs = jran.uniform(tobs_key, minval=0, maxval=20, shape=(ngals,))
        ftpt0 = ftpt0_cens._ftpt0_kernel(
            ftpt0_cens.DEFAULT_FTPT0_PARAMS, lgm_obs, t_obs
        )
        assert np.all(ftpt0 >= 0)
        assert np.all(ftpt0 <= 1)


def test_ftpt0_kernel_is_bounded_random_params():
    ntests = 20
    ngals = 2_000

    ran_key = jran.key(0)

    for __ in range(ntests):
        ran_key, lgm_key, tobs_key, param_key = jran.split(ran_key, 4)
        lgm_obs = jran.uniform(lgm_key, minval=5, maxval=20, shape=(ngals,))
        t_obs = jran.uniform(tobs_key, minval=0, maxval=20, shape=(ngals,))
        u_params = jran.uniform(
            param_key,
            minval=-100,
            maxval=100,
            shape=(len(ftpt0_cens.DEFAULT_FTPT0_PARAMS),),
        )
        u_params = ftpt0_cens.FTPT0_UParams(*u_params)
        params = ftpt0_cens.get_bounded_ftpt0_params(u_params)
        ftpt0 = ftpt0_cens._ftpt0_kernel(params, lgm_obs, t_obs)
        assert np.all(ftpt0 >= 0)
        assert np.all(ftpt0 <= 1)
        assert np.any(ftpt0 > 0)
        assert np.any(ftpt0 < 1)

        # Check monotonic
        lgmarr = np.linspace(5, 20, 100)
        tobsarr = np.zeros_like(lgmarr) + t_obs[0]
        ftpt0arr = ftpt0_cens._ftpt0_kernel(params, lgmarr, tobsarr)
        msk = (ftpt0arr > ftpt0_cens.FTPT0_BOUNDS[0]) & (
            ftpt0arr < ftpt0_cens.FTPT0_BOUNDS[1]
        )
        delta_ftpt0arr = np.diff(ftpt0arr[msk])
        assert np.all(delta_ftpt0arr > 0), (params, t_obs[0])
