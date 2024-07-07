"""
"""

import numpy as np
from jax import random as jran

from .. import tp_pdf_sats as tps

TOL = 1e-3


def test_mc_utp_pdf():
    ran_key = jran.key(0)
    lgmparr = np.linspace(10, 15, 100)
    tobsarr = np.linspace(2, 12, 100)
    args = (ran_key, lgmparr, tobsarr)
    utp = tps.mc_utp_pdf(tps.DEFAULT_UTP_SATPOP_PARAMS, *args)
    assert utp.shape == (100,)
    assert np.all(np.isfinite(utp))
    assert np.all(utp >= 0)
    assert np.all(utp <= 1)


def test_get_utp_loc_kern2():
    lgmparr = np.linspace(10, 15, 100)
    tobsarr = np.linspace(2, 12, 100)
    utp_loc = tps._get_utp_loc_kern(tps.DEFAULT_UTP_SATPOP_PARAMS, lgmparr, tobsarr)
    assert utp_loc.shape == (100,)
    assert np.all(np.isfinite(utp_loc))
    assert np.all(utp_loc >= 0)
    assert np.all(utp_loc <= 1)


def test_get_utp_scale_kern2():
    lgmparr = np.linspace(10, 15, 100)
    tobsarr = np.linspace(2, 12, 100)
    utp_scale = tps._get_utp_scale_kern(tps.DEFAULT_UTP_SATPOP_PARAMS, lgmparr, tobsarr)
    assert utp_scale.shape == (100,)
    assert np.all(np.isfinite(utp_scale))
    assert np.all(utp_scale >= 0)
    assert np.all(utp_scale <= 1)


def test_param_u_param_names_propagate_properly():
    gen = zip(
        tps.DEFAULT_UTP_SATPOP_U_PARAMS._fields, tps.DEFAULT_UTP_SATPOP_PARAMS._fields
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = tps.get_bounded_utp_satpop_params(
        tps.DEFAULT_UTP_SATPOP_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        tps.DEFAULT_UTP_SATPOP_PARAMS._fields
    )

    inferred_default_u_params = tps.get_unbounded_utp_satpop_params(
        tps.DEFAULT_UTP_SATPOP_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        tps.DEFAULT_UTP_SATPOP_U_PARAMS._fields
    )


def test_get_bounded_utp_satpop_params_fails_when_passing_params():
    try:
        tps.get_bounded_utp_satpop_params(tps.DEFAULT_UTP_SATPOP_PARAMS)
        raise NameError("get_bounded_utp_satpop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_utp_satpop_params_fails_when_passing_u_params():
    try:
        tps.get_unbounded_utp_satpop_params(tps.DEFAULT_UTP_SATPOP_U_PARAMS)
        raise NameError("get_unbounded_tpk_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        tps.DEFAULT_UTP_SATPOP_PARAMS,
        tps.get_bounded_utp_satpop_params(tps.DEFAULT_UTP_SATPOP_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = tps.get_bounded_utp_satpop_params(
        tps.get_unbounded_utp_satpop_params(tps.DEFAULT_UTP_SATPOP_PARAMS)
    )
    assert np.allclose(tps.DEFAULT_UTP_SATPOP_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in tps.DEFAULT_UTP_SATPOP_PDICT.keys():
        val = getattr(tps.DEFAULT_UTP_SATPOP_PARAMS, key)
        bound = getattr(tps.UTP_SATPOP_BOUNDS, key)
        assert bound[0] < val < bound[1]
