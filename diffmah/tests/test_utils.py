"""
"""
import numpy as np
from jax import jit as jax_jit
from jax import numpy as jax_np
from jax import value_and_grad

from ..utils import (
    get_cholesky_from_params,
    jax_adam_wrapper,
    jax_inverse_sigmoid,
    jax_sigmoid,
)


def test_inverse_sigmoid_actually_inverts():
    """"""
    x0, k, ylo, yhi = 0, 5, 1, 0
    xarr = np.linspace(-1, 1, 100)
    yarr = np.array(jax_sigmoid(xarr, x0, k, ylo, yhi))
    xarr2 = np.array(jax_inverse_sigmoid(yarr, x0, k, ylo, yhi))
    assert np.allclose(xarr, xarr2, rtol=1e-3)


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


def test_get_cholesky_from_params1():
    ndim = 2
    nparams = int(0.5 * ndim * (ndim + 1))
    params = np.random.uniform(0, 1, nparams)
    chol = get_cholesky_from_params(params)
    row0 = (params[0], 0)
    row1 = (params[2], params[1])
    correct_chol = np.array((row0, row1))
    assert np.allclose(chol, correct_chol)

    ndim = 3
    nparams = int(0.5 * ndim * (ndim + 1))
    params = np.random.uniform(0, 1, nparams)
    chol = get_cholesky_from_params(params)
    row0 = (params[0], 0, 0)
    row1 = (params[3], params[1], 0)
    row2 = (params[4], params[5], params[2])
    correct_chol = np.array((row0, row1, row2))
    assert np.allclose(chol, correct_chol)

    ndim = 4
    nparams = int(0.5 * ndim * (ndim + 1))
    params = np.random.uniform(0, 1, nparams)
    chol = get_cholesky_from_params(params)
    row0 = (params[0], 0, 0, 0)
    row1 = (params[4], params[1], 0, 0)
    row2 = (params[5], params[6], params[2], 0)
    row3 = (params[7], params[8], params[9], params[3])
    correct_chol = np.array((row0, row1, row2, row3))
    assert np.allclose(chol, correct_chol)
