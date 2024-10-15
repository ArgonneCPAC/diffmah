"""
"""

import numpy as np

from .. import logm0_c1_late as c1k

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(
        c1k.DEFAULT_LGM0POP_C1_U_PARAMS._fields, c1k.DEFAULT_LGM0POP_C1_PARAMS._fields
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = c1k.get_bounded_c1_params(c1k.DEFAULT_LGM0POP_C1_U_PARAMS)
    assert set(inferred_default_params._fields) == set(
        c1k.DEFAULT_LGM0POP_C1_PARAMS._fields
    )

    inferred_default_u_params = c1k.get_unbounded_c1_params(
        c1k.DEFAULT_LGM0POP_C1_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        c1k.DEFAULT_LGM0POP_C1_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        c1k.get_bounded_c1_params(c1k.DEFAULT_LGM0POP_C1_PARAMS)
        raise NameError("get_bounded_c0_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        c1k.get_unbounded_c0_params(c1k.DEFAULT_LGM0POP_C1_U_PARAMS)
        raise NameError("get_unbounded_c1_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        c1k.DEFAULT_LGM0POP_C1_PARAMS,
        c1k.get_bounded_c1_params(c1k.DEFAULT_LGM0POP_C1_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = c1k.get_bounded_c1_params(
        c1k.get_unbounded_c1_params(c1k.DEFAULT_LGM0POP_C1_PARAMS)
    )
    assert np.allclose(c1k.DEFAULT_LGM0POP_C1_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in c1k.DEFAULT_LGM0POP_C1_PARAMS._fields:
        val = getattr(c1k.DEFAULT_LGM0POP_C1_PARAMS, key)
        bound = getattr(c1k.LGM0POP_C1_BOUNDS, key)
        assert bound[0] < val < bound[1], key
