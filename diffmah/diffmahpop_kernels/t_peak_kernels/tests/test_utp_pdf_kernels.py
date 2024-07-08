"""
"""

import numpy as np
from jax import random as jran

from ....bfgs_wrapper import diffmah_fitter
from .. import utp_pdf_kernels as tpk


def test_tp_pdf_kern():
    x = 0.5
    pdf = tpk._tp_pdf_kern(x, tpk.DEFAULT_UTP_PARAMS)
    assert np.all(pdf > 0)

    xarr = np.linspace(-2, 2, 500)
    pdf = tpk._tp_pdf_kern(xarr, tpk.DEFAULT_UTP_PARAMS)

    assert np.allclose(pdf[xarr <= 0], 0.0)
    assert np.allclose(pdf[xarr >= 1], 0.0)


def test_mc_tp_pdf_singlesat():
    loc, scale = 0.5, 0.5
    params = (loc, scale)

    ran_key = jran.key(0)
    x_tp = tpk.mc_tp_pdf_singlesat(ran_key, params)
    assert np.all(x_tp > 0)
    assert np.all(x_tp < 1)
    assert x_tp.shape == ()


def test_mc_tp_pdf_satpop():
    n_sats = int(1e4)
    ZZ = np.zeros(n_sats)

    loc, scale = 0.5, 0.5
    params = tpk.UTP_Params(loc + ZZ, scale + ZZ)
    ran_key = jran.key(0)
    x_tp = tpk.mc_tp_pdf_satpop(ran_key, params)
    assert x_tp.shape == (n_sats,)
    assert np.all(x_tp >= 0.0)
    assert np.all(x_tp <= 1.0)


TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(tpk.DEFAULT_UTP_U_PARAMS._fields, tpk.DEFAULT_UTP_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = tpk.get_bounded_utp_params(tpk.DEFAULT_UTP_U_PARAMS)
    assert set(inferred_default_params._fields) == set(tpk.DEFAULT_UTP_PARAMS._fields)

    inferred_default_u_params = tpk.get_unbounded_utp_params(tpk.DEFAULT_UTP_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        tpk.DEFAULT_UTP_U_PARAMS._fields
    )


def test_get_bounded_utp_params_fails_when_passing_params():
    try:
        tpk.get_bounded_utp_params(tpk.DEFAULT_UTP_PARAMS)
        raise NameError("get_bounded_utp_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_utp_params_fails_when_passing_u_params():
    try:
        tpk.get_unbounded_tpk_params(tpk.DEFAULT_UTP_U_PARAMS)
        raise NameError("get_unbounded_tpk_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    assert np.allclose(
        tpk.DEFAULT_UTP_PARAMS,
        tpk.get_bounded_utp_params(tpk.DEFAULT_UTP_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = tpk.get_bounded_utp_params(
        tpk.get_unbounded_utp_params(tpk.DEFAULT_UTP_PARAMS)
    )
    assert np.allclose(tpk.DEFAULT_UTP_PARAMS, inferred_default_params, rtol=TOL)


def test_default_params_are_in_bounds():
    for key in tpk.DEFAULT_UTP_PDICT.keys():
        val = getattr(tpk.DEFAULT_UTP_PARAMS, key)
        bound = getattr(tpk.UTP_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_utp_param_fitter():
    tp_by_tobs_hist_binmids = np.linspace(0.01, 0.99, 1000)
    p_target = tpk.DEFAULT_UTP_PARAMS
    pdf_target = tpk._tp_pdf_kern(tp_by_tobs_hist_binmids, p_target)
    loss_data = tp_by_tobs_hist_binmids, pdf_target
    u_p_init = tpk.DEFAULT_UTP_U_PARAMS
    u_p_init = u_p_init._replace(u_utp_loc=0.7, u_utp_scale=0.2)

    loss_init, grads_init = tpk.loss_and_grads_kern(u_p_init, loss_data)
    for grad in grads_init:
        assert np.all(np.isfinite(grad))
    assert loss_init > 0

    args = (tpk.loss_and_grads_kern, u_p_init, loss_data)

    u_p_best, loss_best, fit_terminates, code_used = diffmah_fitter(*args)
    u_p_best = tpk.UTP_UParams(*u_p_best)
    p_best = tpk.get_bounded_utp_params(u_p_best)

    assert np.allclose(p_best.utp_loc, p_target.utp_loc, rtol=0.01)
    assert np.allclose(p_best.utp_scale, p_target.utp_scale, rtol=0.01)
    assert fit_terminates
    assert code_used == 0
    assert loss_best < loss_init / 2