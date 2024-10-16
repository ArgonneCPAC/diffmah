""""""

import numpy as np
from jax import random as jran

from ...tests.test_utils import _enforce_is_cov
from .. import covariance_kernels as ck


def test_param_u_param_names_propagate_properly():
    gen = zip(ck.DEFAULT_COV_U_PARAMS._fields, ck.DEFAULT_COV_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = ck.get_bounded_cov_params(ck.DEFAULT_COV_U_PARAMS)
    assert set(inferred_default_params._fields) == set(ck.DEFAULT_COV_PARAMS._fields)

    inferred_default_u_params = ck.get_unbounded_cov_params(ck.DEFAULT_COV_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        ck.DEFAULT_COV_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        ck.get_bounded_cov_params(ck.DEFAULT_COV_PARAMS)
        raise NameError("get_bounded_cov_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        ck.get_unbounded_cov_params(ck.DEFAULT_COV_U_PARAMS)
        raise NameError("get_unbounded_cov_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    ran_key = jran.key(0)
    n_tests = 100
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        n_p = len(ck.DEFAULT_COV_PARAMS)
        u_p = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_p,))
        u_p = ck.CovUParams(*u_p)
        p = ck.get_bounded_cov_params(u_p)
        u_p2 = ck.get_unbounded_cov_params(p)
        for x, y in zip(u_p, u_p2):
            assert np.allclose(x, y, rtol=0.01)


def test_default_params_are_in_bounds():
    for key in ck.DEFAULT_COV_PARAMS._fields:
        val = getattr(ck.DEFAULT_COV_PARAMS, key)
        bound = getattr(ck.COV_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_covariances_are_always_covariances():
    lgmarr = np.linspace(10, 15, 20)
    ran_key = jran.key(0)
    npars = len(ck.DEFAULT_COV_PARAMS)
    ntests = 200
    for __ in range(ntests):
        ran_key, test_key = jran.split(ran_key, 2)
        u_p = jran.uniform(test_key, minval=-1000, maxval=1000, shape=(npars,))
        u_params = ck.CovUParams(*u_p)
        cov_params = ck.get_bounded_cov_params(u_params)

        for lgm in lgmarr:
            cov = ck._get_diffmahpop_cov(cov_params, lgm)
            assert cov.shape == (4, 4)
            _enforce_is_cov(cov)
