"""
"""

import numpy as np
from jax.example_libraries import optimizers as jax_opt


def jax_adam_wrapper(
    loss_and_grad_func,
    params_init,
    loss_data,
    n_step,
    n_warmup=0,
    step_size=0.01,
    warmup_n_step=50,
    warmup_step_size=None,
    tol=0.0,
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_and_grad_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar and its gradients

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    n_warmup : int, optional
        Number of warmup iterations. At the end of the warmup, the best-fit parameters
        are used as input parameters to the final burn. Default is zero.

    warmup_n_step : int, optional
        Number of Adam steps to take during warmup. Default is 50.

    warmup_step_size : float, optional
        Step size to use during warmup phase. Default is 5*step_size.

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01.

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    fit_terminates : int
        0 if NaN or inf is encountered by the fitter, causing termination before n_step
        1 for a fit that terminates with no such problems

    """
    if warmup_step_size is None:
        warmup_step_size = 5 * step_size

    p_init = np.copy(params_init)
    loss_init = float("inf")
    for i in range(n_warmup):
        fit_results = _jax_adam_wrapper(
            loss_and_grad_func,
            p_init,
            loss_data,
            warmup_n_step,
            step_size=warmup_step_size,
            tol=tol,
        )
        p_init = fit_results[0]
        loss_init = fit_results[1]

    if np.all(np.isfinite(p_init)):
        p0 = p_init
    else:
        p0 = params_init

    if loss_init > tol:
        fit_results = _jax_adam_wrapper(
            loss_and_grad_func, p0, loss_data, n_step, step_size=step_size, tol=tol
        )

    if len(fit_results[2]) < n_step:
        fit_terminates = 0
    else:
        fit_terminates = 1
    return (*fit_results, fit_terminates)


def _jax_adam_wrapper(
    loss_and_grad_func, params_init, loss_data, n_step, step_size=0.01, tol=0.0
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_and_grad_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar loss and its gradients

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    """
    loss_collector = []
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)
    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    for istep in range(n_step):
        p = np.array(get_params(opt_state))

        loss, grads = loss_and_grad_func(p, loss_data)
        loss_collector.append(loss)

        no_nan_params = np.all(np.isfinite(p))
        no_nan_loss = np.isfinite(loss)
        no_nan_grads = np.all(np.isfinite(grads))
        has_nans = ~no_nan_params | ~no_nan_loss | ~no_nan_grads
        if has_nans:
            if istep > 0:
                indx_best = np.nanargmin(loss_collector[:istep])
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_collector[indx_best]
            else:
                best_fit_params = np.copy(p)
                best_fit_loss = 999.99
            return (
                best_fit_params,
                best_fit_loss,
                np.array(loss_collector[:istep]),
                params_arr[:istep, :],
            )
        else:
            params_arr[istep, :] = p
            opt_state = opt_update(istep, grads, opt_state)
            if loss < tol:
                best_fit_params = p
                loss_arr = np.array(loss_collector)
                return best_fit_params, loss, loss_arr, params_arr

    loss_arr = np.array(loss_collector)
    indx_best = np.nanargmin(loss_arr)
    best_fit_params = params_arr[indx_best]
    loss = loss_arr[indx_best]

    return best_fit_params, loss, loss_arr, params_arr
