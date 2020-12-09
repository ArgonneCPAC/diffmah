"""
"""
import pytest
import functools
from jax import numpy as jax_np
from jax import jit as jax_jit
from jax import value_and_grad
import numpy as np
from ..quenching_history import mean_log_main_sequence_fraction
from ..quenching_history import _log_main_sequence_fraction
from ..quenching_history import _mean_log_main_sequence_fraction
from ..quenching_history import MEAN_Q_PARAMS


def test_mean_log_main_sequence_fraction1():
    lgtarr = np.linspace(0, 1.15, 500)
    for lgm in np.arange(10, 15, 0.5):
        lgp_ms = mean_log_main_sequence_fraction(lgtarr, lgm)
        assert np.all(lgp_ms <= 0)
        assert lgp_ms.size == lgtarr.size


@pytest.mark.xfail
def test_ms_frac_is_differentiable():
    @functools.partial(jax_jit, static_argnums=(1,))
    def mse_loss(params, data):
        fms_x0, fms_yhi = params
        logt, target = data
        log_prob_ms = _log_main_sequence_fraction(fms_x0, fms_yhi, logt)
        diff = target - log_prob_ms
        return jax_np.sum(diff) / diff.size

    npts = 250
    logtmp = np.log10(13.85)
    logt = np.linspace(-1, logtmp, npts)
    params = np.array((1.0, -0.5))
    target = _log_main_sequence_fraction(logt, *params)
    data = logt, target
    loss_init, grads = value_and_grad(mse_loss, argnums=0)(params, data)
    params_new = params - 0.05 * grads
    loss_new, grads = value_and_grad(mse_loss, argnums=0)(params_new, data)
    assert loss_new < loss_init


@pytest.mark.xfail
def test_mean_ms_frac_is_differentiable():
    @functools.partial(jax_jit, static_argnums=(1,))
    def mse_loss(params, mse_loss_data):
        logmp, logt, target = mse_loss_data
        log_prob_ms = _mean_log_main_sequence_fraction(logt, logmp, *params)
        diff = target - log_prob_ms
        return jax_np.sum(diff) / diff.size

    logmp = 12
    npts = 250
    logtmp = np.log10(13.85)
    logt = np.linspace(-1, logtmp, npts)

    params_fid = np.array(list(MEAN_Q_PARAMS.values()))
    target = _mean_log_main_sequence_fraction(logt, logmp, *params_fid)
    params_init = params_fid * 0.9 - 0.05
    data = logmp, logt, target
    loss_init, grads = value_and_grad(mse_loss, argnums=0)(params_init, data)
    params_new = params_init - 0.02 * grads
    loss_new, grads = value_and_grad(mse_loss, argnums=0)(params_new, data)
    assert loss_new < loss_init
