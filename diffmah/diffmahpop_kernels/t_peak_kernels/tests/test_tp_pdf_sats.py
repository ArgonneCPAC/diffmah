"""
"""

import numpy as np
import pytest
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from .. import tp_pdf_sats as tps
from .. import utp_pdf_kernels

TOL = 1e-3


def test_mc_tpeak_pdf():
    ran_key = jran.key(0)
    ran_key, pred_key, lgm_key, tobs_key = jran.split(ran_key, 4)
    nsats = int(1e4)
    lgmparr = jran.uniform(lgm_key, minval=10, maxval=15, shape=(nsats,))
    tobsarr = jran.uniform(tobs_key, minval=2, maxval=12, shape=(nsats,))
    args = (pred_key, lgmparr, tobsarr)
    tpeak = tps.mc_tpeak_sats(tps.DEFAULT_UTP_SATPOP_PARAMS, *args)
    assert tpeak.shape == (nsats,)
    assert np.all(np.isfinite(tpeak))
    assert np.all(tpeak > tobsarr * utp_pdf_kernels.X_MIN)
    assert np.all(tpeak < tobsarr)

    ntests = 10
    npars = len(tps.DEFAULT_UTP_SATPOP_U_PARAMS)
    for __ in range(ntests):
        ran_key, test_key = jran.split(ran_key, 2)

        pred_key, u_p_key, lgm_key, tobs_key = jran.split(test_key, 4)
        nsats = int(1e5)
        lgmparr = jran.uniform(lgm_key, minval=10, maxval=15, shape=(nsats,))
        tobsarr = jran.uniform(tobs_key, minval=2, maxval=12, shape=(nsats,))

        u_p = jran.uniform(u_p_key, minval=-100, maxval=100, shape=(npars,))
        u_p = tps.DEFAULT_UTP_SATPOP_U_PARAMS._make(u_p)
        params = tps.get_bounded_utp_satpop_params(u_p)

        args = (pred_key, lgmparr, tobsarr)
        tpeak = tps.mc_tpeak_sats(params, *args)
        assert tpeak.shape == (nsats,)
        assert np.all(np.isfinite(tpeak))
        assert np.all(tpeak > tobsarr * utp_pdf_kernels.X_MIN)
        assert np.all(tpeak < tobsarr)


@pytest.mark.xfail
def test_mc_tp_pdf_satpop_is_differentiable():
    ran_key = jran.key(0)

    n_tests = 10

    for __ in range(n_tests):

        lgm_key, t_key, ran_key = jran.split(ran_key, 3)
        lgmparr = jran.uniform(lgm_key, minval=11.0, maxval=14.0, shape=(200,))
        tobsarr = jran.uniform(t_key, minval=3.0, maxval=13.0, shape=(200,))
        t_peak_target_sample = tps.mc_utp_pdf(
            tps.DEFAULT_UTP_SATPOP_PARAMS, ran_key, lgmparr, tobsarr
        )
        target_mean_tp = jnp.mean(t_peak_target_sample)

        @jjit
        def _mse(x, y):
            d = y - x
            return jnp.mean(d * d)

        @jjit
        def _mean_tp_loss(u_params, pred_key):
            params = tps.get_bounded_utp_satpop_params(u_params)
            t_peak_sample = tps.mc_utp_pdf(params, pred_key, lgmparr, tobsarr)
            pred_mean_tp = jnp.mean(t_peak_sample)
            return _mse(pred_mean_tp, target_mean_tp)

        mean_tp_loss_and_grad = value_and_grad(_mean_tp_loss)

        n_params = len(tps.DEFAULT_UTP_SATPOP_PARAMS)
        u_p_key, loss_pred_key = jran.split(ran_key, 2)
        uran = jran.uniform(u_p_key, minval=-10, maxval=10, shape=(n_params,))
        u_p_init = tps.DEFAULT_UTP_SATPOP_U_PARAMS._make(
            uran + np.array(tps.DEFAULT_UTP_SATPOP_U_PARAMS)
        )

        loss, grads = mean_tp_loss_and_grad(u_p_init, loss_pred_key)
        assert np.all(np.isfinite(loss))
        assert np.all(np.isfinite(grads))


def test_mc_utp_pdf():
    ran_key = jran.key(0)
    lgmparr = np.linspace(10, 15, 100)
    tobsarr = np.linspace(2, 12, 100)
    args = (ran_key, lgmparr, tobsarr)
    utp = tps.mc_utp_pdf(tps.DEFAULT_UTP_SATPOP_PARAMS, *args)
    assert utp.shape == (100,)
    assert np.all(np.isfinite(utp))
    assert np.all(utp > 0)
    assert np.all(utp < 1)


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
