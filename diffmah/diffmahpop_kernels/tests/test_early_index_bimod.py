"""
"""

import numpy as np

from .. import early_index_bimod as eip

TOL = 1e-3


def test_param_u_param_names_propagate_properly():
    gen = zip(
        eip.DEFAULT_EARLY_INDEX_U_PARAMS._fields,
        eip.DEFAULT_EARLY_INDEX_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = eip.get_bounded_early_index_params(
        eip.DEFAULT_EARLY_INDEX_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        eip.DEFAULT_EARLY_INDEX_PARAMS._fields
    )

    inferred_default_u_params = eip.get_unbounded_early_index_params(
        eip.DEFAULT_EARLY_INDEX_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        eip.DEFAULT_EARLY_INDEX_U_PARAMS._fields
    )


def test_get_bounded_early_index_params_fails_when_passing_params():
    try:
        eip.get_bounded_early_index_params(eip.DEFAULT_EARLY_INDEX_PARAMS)
        raise NameError("get_bounded_early_index_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_early_index_params_fails_when_passing_u_params():
    try:
        eip.get_unbounded_early_index_params(eip.DEFAULT_EARLY_INDEX_U_PARAMS)
        raise NameError("get_unbounded_early_index_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        eip.DEFAULT_EARLY_INDEX_PARAMS,
        eip.get_bounded_early_index_params(eip.DEFAULT_EARLY_INDEX_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = eip.get_bounded_early_index_params(
        eip.get_unbounded_early_index_params(eip.DEFAULT_EARLY_INDEX_PARAMS)
    )
    assert np.allclose(
        eip.DEFAULT_EARLY_INDEX_PARAMS, inferred_default_params, rtol=TOL
    )


def test_default_params_are_in_bounds():
    for key in eip.DEFAULT_EARLY_INDEX_PARAMS._fields:
        val = getattr(eip.DEFAULT_EARLY_INDEX_PARAMS, key)
        bound = getattr(eip.EARLY_INDEX_PBOUNDS, key)
        assert bound[0] < val < bound[1]
