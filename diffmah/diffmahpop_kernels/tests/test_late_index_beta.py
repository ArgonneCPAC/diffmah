"""
"""

import numpy as np

from .. import late_index_beta as lip

TOL = 1e-3


def test_param_u_param_names_propagate_properly():
    gen = zip(
        lip.DEFAULT_LATE_INDEX_U_PARAMS._fields,
        lip.DEFAULT_LATE_INDEX_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = lip.get_bounded_late_index_params(
        lip.DEFAULT_LATE_INDEX_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        lip.DEFAULT_LATE_INDEX_PARAMS._fields
    )

    inferred_default_u_params = lip.get_unbounded_late_index_params(
        lip.DEFAULT_LATE_INDEX_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        lip.DEFAULT_LATE_INDEX_U_PARAMS._fields
    )


def test_get_bounded_late_index_params_fails_when_passing_params():
    try:
        lip.get_bounded_late_index_params(lip.DEFAULT_LATE_INDEX_PARAMS)
        raise NameError("get_bounded_late_index_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_late_index_params_fails_when_passing_u_params():
    try:
        lip.get_unbounded_late_index_params(lip.DEFAULT_LATE_INDEX_U_PARAMS)
        raise NameError("get_unbounded_late_index_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        lip.DEFAULT_LATE_INDEX_PARAMS,
        lip.get_bounded_late_index_params(lip.DEFAULT_LATE_INDEX_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = lip.get_bounded_late_index_params(
        lip.get_unbounded_late_index_params(lip.DEFAULT_LATE_INDEX_PARAMS)
    )
    assert np.allclose(lip.DEFAULT_LATE_INDEX_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in lip.DEFAULT_LATE_INDEX_PARAMS._fields:
        val = getattr(lip.DEFAULT_LATE_INDEX_PARAMS, key)
        bound = getattr(lip.LATE_INDEX_PBOUNDS, key)
        assert bound[0] < val < bound[1]
