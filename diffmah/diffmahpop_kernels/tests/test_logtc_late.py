"""
"""

import numpy as np

from .. import logtc_late as tcp

TOL = 1e-3


def test_param_u_param_names_propagate_properly():
    gen = zip(
        tcp.DEFAULT_LOGTC_U_PARAMS._fields,
        tcp.DEFAULT_LOGTC_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = tcp.get_bounded_logtc_params(tcp.DEFAULT_LOGTC_U_PARAMS)
    assert set(inferred_default_params._fields) == set(tcp.DEFAULT_LOGTC_PARAMS._fields)

    inferred_default_u_params = tcp.get_unbounded_logtc_params(tcp.DEFAULT_LOGTC_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        tcp.DEFAULT_LOGTC_U_PARAMS._fields
    )


def test_get_bounded_logtc_params_fails_when_passing_params():
    try:
        tcp.get_bounded_logtc_params(tcp.DEFAULT_LOGTC_PARAMS)
        raise NameError("get_bounded_logtc_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_logtc_params_fails_when_passing_u_params():
    try:
        tcp.get_unbounded_logtc_params(tcp.DEFAULT_LOGTC_U_PARAMS)
        raise NameError("get_unbounded_logtc_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        tcp.DEFAULT_LOGTC_PARAMS,
        tcp.get_bounded_logtc_params(tcp.DEFAULT_LOGTC_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = tcp.get_bounded_logtc_params(
        tcp.get_unbounded_logtc_params(tcp.DEFAULT_LOGTC_PARAMS)
    )
    assert np.allclose(tcp.DEFAULT_LOGTC_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in tcp.DEFAULT_LOGTC_PARAMS._fields:
        val = getattr(tcp.DEFAULT_LOGTC_PARAMS, key)
        bound = getattr(tcp.LOGTC_PBOUNDS, key)
        assert bound[0] < val < bound[1]
