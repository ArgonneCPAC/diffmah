"""
"""

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from .. import tp_pdf_sats as tps
from .. import utp_pdf_kernels as utpk

TOL = 1e-3


def test_mc_tp_pdf_satpop_is_differentiable():
    ran_key = jran.key(0)

    n_tests = 5

    for __ in range(n_tests):

        lgm_key, t_key, ran_key = jran.split(ran_key, 3)
        lgmparr = jran.uniform(lgm_key, minval=11.0, maxval=14.0, shape=(200,))
        tobsarr = jran.uniform(t_key, minval=3.0, maxval=13.0, shape=(200,))
        t_peak_target_sample = tps.mc_utp_pdf(
            tps.DEFAULT_TP_SATS_PARAMS, ran_key, lgmparr, tobsarr
        )
        target_mean_tp = jnp.mean(t_peak_target_sample)

        @jjit
        def _mse(x, y):
            d = y - x
            return jnp.mean(d * d)

        @jjit
        def _mean_tp_loss(u_params, pred_key):
            params = tps.get_bounded_tp_sat_params(u_params)
            t_peak_sample = tps.mc_utp_pdf(params, pred_key, lgmparr, tobsarr)
            pred_mean_tp = jnp.mean(t_peak_sample)
            return _mse(pred_mean_tp, target_mean_tp)

        mean_tp_loss_and_grad = value_and_grad(_mean_tp_loss)

        n_params = len(tps.DEFAULT_TP_SATS_PARAMS)
        u_p_key, loss_pred_key = jran.split(ran_key, 2)
        uran = jran.uniform(u_p_key, minval=-10, maxval=10, shape=(n_params,))
        u_p_init = tps.DEFAULT_TP_SATS_U_PARAMS._make(
            uran + np.array(tps.DEFAULT_TP_SATS_U_PARAMS)
        )

        loss, grads = mean_tp_loss_and_grad(u_p_init, loss_pred_key)
        assert np.all(np.isfinite(loss))
        assert np.all(np.isfinite(grads))


def test_mc_utp_pdf_is_bounded_default_model():
    ran_key = jran.key(0)
    lgm_key, t_key = jran.split(ran_key, 2)
    lgmparr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(200,))
    tobsarr = jran.uniform(t_key, minval=1, maxval=15, shape=(200,))
    args = (ran_key, lgmparr, tobsarr)
    utp = tps.mc_utp_pdf(tps.DEFAULT_TP_SATS_PARAMS, *args)
    assert utp.shape == lgmparr.shape
    assert np.all(np.isfinite(utp))
    assert np.all(utp > utpk.X_MIN)
    assert np.all(utp < utpk.X_MAX)


def test_mc_utp_pdf_is_bounded_random_model():
    ran_key = jran.key(0)
    n_tests = 100
    n_params = len(tps.DEFAULT_TP_SATS_PARAMS)
    for __ in range(n_tests):
        lgm_key, t_key, up_key, ran_key = jran.split(ran_key, 4)
        uran = jran.uniform(up_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = tps.DEFAULT_TP_SATS_U_PARAMS._make(uran)
        params = tps.get_bounded_tp_sat_params(u_params)
        lgmparr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(2_000,))
        tobsarr = jran.uniform(t_key, minval=1, maxval=15, shape=(2_000,))
        args = (ran_key, lgmparr, tobsarr)
        utp = tps.mc_utp_pdf(params, *args)
        assert utp.shape == lgmparr.shape
        assert np.all(np.isfinite(utp))
        assert np.all(utp > utpk.X_MIN)
        assert np.all(utp < utpk.X_MAX)


def test_get_utp_loc_returns_in_bounds_default_model():
    ran_key = jran.key(0)
    lgm_key, t_key = jran.split(ran_key, 2)
    lgmparr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(2_000,))
    tobsarr = jran.uniform(t_key, minval=1, maxval=15, shape=(2_000,))
    utp_loc = tps._get_utp_loc(tps.DEFAULT_TP_SATS_PARAMS, lgmparr, tobsarr)
    assert utp_loc.shape == lgmparr.shape
    assert np.all(np.isfinite(utp_loc))
    assert np.all(utp_loc > utpk.UTP_PBOUNDS.utp_loc[0])
    assert np.all(utp_loc < utpk.UTP_PBOUNDS.utp_loc[1])


def test_get_utp_loc_returns_in_bounds_random_model():
    ran_key = jran.key(0)
    n_tests = 100
    n_params = len(tps.DEFAULT_TP_SATS_PARAMS)
    for __ in range(n_tests):
        lgm_key, t_key, up_key, ran_key = jran.split(ran_key, 4)
        uran = jran.uniform(up_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = tps.DEFAULT_TP_SATS_U_PARAMS._make(uran)
        params = tps.get_bounded_tp_sat_params(u_params)
        lgmparr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(2_000,))
        tobsarr = jran.uniform(t_key, minval=1, maxval=15, shape=(2_000,))
        utp_loc = tps._get_utp_loc(params, lgmparr, tobsarr)
        assert utp_loc.shape == lgmparr.shape
        assert np.all(np.isfinite(utp_loc))
        assert np.all(utp_loc > utpk.UTP_PBOUNDS.utp_loc[0])
        assert np.all(utp_loc < utpk.UTP_PBOUNDS.utp_loc[1])


def test_get_utp_scale_returns_in_bounds_default_model():
    ran_key = jran.key(0)
    lgm_key, t_key = jran.split(ran_key, 2)
    lgmparr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(2_000,))
    tobsarr = jran.uniform(t_key, minval=1, maxval=15, shape=(2_000,))
    utp_scale = tps._get_utp_scale(tps.DEFAULT_TP_SATS_PARAMS, lgmparr, tobsarr)
    assert utp_scale.shape == lgmparr.shape
    assert np.all(np.isfinite(utp_scale))
    assert np.all(utp_scale > utpk.UTP_PBOUNDS.utp_scale[0])
    assert np.all(utp_scale < utpk.UTP_PBOUNDS.utp_scale[1])


def test_get_utp_scale_returns_in_bounds_random_model():
    ran_key = jran.key(0)
    n_tests = 100
    n_params = len(tps.DEFAULT_TP_SATS_PARAMS)
    for __ in range(n_tests):
        lgm_key, t_key, up_key, ran_key = jran.split(ran_key, 4)
        uran = jran.uniform(up_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = tps.DEFAULT_TP_SATS_U_PARAMS._make(uran)
        params = tps.get_bounded_tp_sat_params(u_params)
        lgmparr = jran.uniform(lgm_key, minval=5, maxval=20, shape=(2_000,))
        tobsarr = jran.uniform(t_key, minval=1, maxval=15, shape=(2_000,))
        utp_scale = tps._get_utp_scale(params, lgmparr, tobsarr)
        assert utp_scale.shape == lgmparr.shape
        assert np.all(np.isfinite(utp_scale))
        assert np.all(utp_scale > utpk.UTP_PBOUNDS.utp_scale[0])
        assert np.all(utp_scale < utpk.UTP_PBOUNDS.utp_scale[1])


def test_get_bounded_tp_sat_params_fails_when_passing_params():
    try:
        tps.get_bounded_tp_sat_params(tps.DEFAULT_TP_SATS_PARAMS)
        raise NameError("get_bounded_tp_sat_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_tp_sat_params_fails_when_passing_params():
    try:
        tps.get_unbounded_tp_sat_params(tps.DEFAULT_TP_SATS_U_PARAMS)
        raise NameError("get_bounded_tp_sat_params should not accept u_params")
    except AttributeError:
        pass


def test_default_params_are_in_bounds():
    for key in tps.DEFAULT_TP_SATS_PARAMS._fields:
        val = getattr(tps.DEFAULT_TP_SATS_PARAMS, key)
        bound = getattr(tps.TP_SATS_BOUNDS, key)
        assert bound[0] < val < bound[1], f"default `{key}` is out of bounds"


def test_param_u_param_inversion():
    assert np.allclose(
        tps.DEFAULT_TP_SATS_PARAMS,
        tps.get_bounded_tp_sat_params(tps.DEFAULT_TP_SATS_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = tps.get_bounded_tp_sat_params(
        tps.get_unbounded_tp_sat_params(tps.DEFAULT_TP_SATS_PARAMS)
    )
    assert np.allclose(tps.DEFAULT_TP_SATS_PARAMS, inferred_default_params, rtol=TOL)


def test_param_u_param_names_propagate_properly():
    gen = zip(tps.DEFAULT_TP_SATS_U_PARAMS._fields, tps.DEFAULT_TP_SATS_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = tps.get_bounded_tp_sat_params(
        tps.DEFAULT_TP_SATS_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        tps.DEFAULT_TP_SATS_PARAMS._fields
    )

    inferred_default_u_params = tps.get_unbounded_tp_sat_params(
        tps.DEFAULT_TP_SATS_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        tps.DEFAULT_TP_SATS_U_PARAMS._fields
    )
