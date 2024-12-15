"""
"""

import numpy as np
from jax import jit as jax_jit
from jax import numpy as jax_np
from jax import value_and_grad

from ..fit_utils import jax_adam_wrapper


def test_jax_adam_wrapper_actually_minimizes_the_loss():
    @jax_jit
    def mse_loss(params, data):
        x = data[0]
        target = 3 * (x - 1)
        a, b = params
        pred = a * (x - b)
        diff = pred - target
        loss = jax_np.sum(diff * diff) / diff.size
        return loss

    @jax_jit
    def mse_loss_and_grad(params, data):
        return value_and_grad(mse_loss, argnums=0)(params, data)

    params_init = np.array((2.75, 0.75))
    x = np.linspace(-1, 1, 50)
    data = (x,)
    loss_init = mse_loss(params_init, data)
    n_step = 100
    params_bestfit, loss_bestfit, loss_arr, params_arr, flag = jax_adam_wrapper(
        mse_loss_and_grad, params_init, data, n_step, step_size=0.01
    )
    assert loss_arr[-1] < loss_init
    assert np.allclose(loss_bestfit, loss_arr[-1], atol=0.001)

    params_correct = [3, 1]
    assert np.allclose(params_bestfit, params_correct, atol=0.01)


def test_jax_adam_wrapper_loss_tol_feature_works():
    @jax_jit
    def mse_loss(params, data):
        x = data[0]
        target = 3 * (x - 1)
        a, b = params
        pred = a * (x - b)
        diff = pred - target
        loss = jax_np.sum(diff * diff) / diff.size
        return loss

    @jax_jit
    def mse_loss_and_grad(params, data):
        return value_and_grad(mse_loss, argnums=0)(params, data)

    params_init = np.array((2.75, 0.75))
    x = np.linspace(-1, 1, 50)

    data = (x,)
    loss_init = mse_loss(params_init, data)
    n_step = 100
    params_bestfit, loss_bestfit0, loss_arr, params_arr, flag = jax_adam_wrapper(
        mse_loss_and_grad, params_init, data, n_step, step_size=0.01, tol=1e-2
    )
    params_bestfit, loss_bestfit1, loss_arr, params_arr, flag = jax_adam_wrapper(
        mse_loss_and_grad, params_init, data, n_step, step_size=0.01, tol=1e-3
    )
    params_bestfit, loss_bestfit2, loss_arr, params_arr, flag = jax_adam_wrapper(
        mse_loss_and_grad, params_init, data, n_step, step_size=0.01, tol=1e-4
    )
    assert loss_bestfit0 <= 1e-2
    assert loss_bestfit1 <= 1e-3
    assert loss_bestfit2 <= 1e-4
    assert loss_bestfit2 < loss_bestfit1 < loss_bestfit0 < loss_init
