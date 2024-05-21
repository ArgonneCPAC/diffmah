"""
Functions in this script fit MAH using an alternating wrapper.
The main minimizer used is scipy's LBFGS algo.
If this does not work (success=False), then it tries minimizing with JAX's ADAM algo.

The main function is minimize_alternate_wrappers.
"""

import numpy as np
from scipy.optimize import minimize

from .utils import jax_adam_wrapper


def scipy_lbfgs_wrapper(val_and_grads, p_init, loss_data):
    """
    Function that runs scipy's LBFGS minimizer.

    Args:
        val_and_grads: function that returns the loss function along with the grads.
        For LBFGS, one does not need to use grads.
        p_init: array of initial values for parameters
        loss_data: Sequence of floats and arrays storing
        whatever data is needed to compute loss_func(params_init, loss_data)
    Returns:
        _res: list of best fit parameters, best fit loss,
        and a boolean whether the fit was successful or not.

    """

    # Define the loss and grad functions from value_and_grads
    # scipy wants them separated
    def loss_func(p_init, loss_data):
        return float(val_and_grads(p_init, loss_data)[0])

    def grad_func(p_init, loss_data):
        return np.array(val_and_grads(p_init, loss_data)[1]).astype(float)

    # run scipy's LBFGS minimizer
    result = minimize(
        loss_func, p_init, method="L-BFGS-B", jac=grad_func, args=(loss_data,)
    )
    _res = [result.x, result.fun, result.success]
    return _res


def minimize_alternate_wrappers(
    val_and_grads, p_init, loss_data, nstep=200, n_warmup=1
):
    """
    Function that runs scipy's LBFGS minimizer.
    If that is not successful, minimize the fit with the ADAM minimizer from JAX.
    Args:
        val_and_grads: function that returns the loss function along with the grads
        For LBFGS, one does not need to use grads.
        p_init: array of initial values for parameters
        loss_data: Sequence of floats and arrays storing
        whatever data is needed to compute loss_func(params_init, loss_data)
        nstep: Number of steps that the ADAM wrapper needs to run for (default = 200)
        n_warmup: The number of warmup steps to use (default = 1)

    Returns:
        _res: list of best fit parameters, best fit loss,
        and a boolean whether the fit was successful or not, and what algo was used.

    """
    _res = scipy_lbfgs_wrapper(val_and_grads, p_init, loss_data)

    # check if LBFGS succeeds. If yes, save those results.
    # Otherwise try the Adam wrapper
    if _res[-1] and _res[1] != 0.0:
        code_used = "LBFGS"
        _res.append(code_used)
        return _res
    else:
        res = jax_adam_wrapper(val_and_grads, p_init, loss_data, nstep, n_warmup)
        p_best, loss_best, loss_arr, params_arr, fit_terminates = res
        code_used = "Adam"
        _res = [p_best, loss_best, fit_terminates, code_used]
        return _res
