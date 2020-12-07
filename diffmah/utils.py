import numpy as np
from jax import numpy as jax_np
from jax import value_and_grad
from jax import jit as jax_jit
from jax.experimental import optimizers as jax_opt
from collections import OrderedDict


def get_1d_arrays(*args, jax_arrays=False):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [jax_np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)

    if jax_arrays:
        result = [jax_np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    else:
        result = [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    return result


def jax_sigmoid(x, x0, k, ylo, yhi):
    """Sigmoid function implemented w/ `jax.numpy.exp`.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the function.
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ylo : float or array-like
        The value as x goes to -infty.
    yhi : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input
    """
    return ylo + (yhi - ylo) / (1 + jax_np.exp(-k * (x - x0)))


def jax_inverse_sigmoid(y, x0, k, ylo, yhi):
    """Sigmoid function implemented w/ `jax.numpy.exp`.

    Parameters
    ----------
    y : float or array-like
        Value of the function
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ylo : float or array-like
        The value as x goes to -infty.
    yhi : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input
    """
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jax_np.log(lnarg) / k


@jax_jit
def _jax_triweight_sigmoid_kernel(y):
    val = jax_np.where(
        y < 3,
        -5 * y ** 7 / 69984
        + 7 * y ** 5 / 2592
        - 35 * y ** 3 / 864
        + 35 * y / 96
        + 1 / 2,
        1,
    )
    return jax_np.where(y > -3, val, 0)


def _enforce_no_extraneous_keywords(defaults, **kwargs):
    unrecognized_params = set(kwargs) - set(defaults)

    if len(unrecognized_params) > 0:
        param = list(unrecognized_params)[0]
        raise KeyError("Unrecognized parameter ``{0}``".format(param))


def _get_param_dict(defaults, strict=False, **kwargs):
    """
    """
    param_dict = OrderedDict(
        [(key, kwargs.get(key, val)) for key, val in defaults.items()]
    )
    if strict:
        _enforce_no_extraneous_keywords(defaults, **kwargs)
    return param_dict


def _get_param_array(defaults, strict=False, dtype="f4", jax_arrays=True, **kwargs):
    """
    """
    param_dict = _get_param_dict(defaults, strict=strict, **kwargs)
    if jax_arrays:
        param_array = jax_np.array(list(param_dict.values())).astype(dtype)
    else:
        param_array = np.array(list(param_dict.values())).astype(dtype)
    return param_array


def jax_adam_wrapper(
    loss_func,
    params_init,
    loss_data,
    n_step,
    n_warmup=0,
    step_size=0.01,
    warmup_n_step=50,
    warmup_step_size=None,
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.

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

    """
    if warmup_step_size is None:
        warmup_step_size = 5 * step_size

    for i in range(n_warmup):
        params_init = _jax_adam_wrapper(
            loss_func, params_init, loss_data, warmup_n_step, step_size=warmup_step_size
        )[0]
    return _jax_adam_wrapper(
        loss_func, params_init, loss_data, n_step, step_size=step_size
    )


def _jax_adam_wrapper(loss_func, params_init, loss_data, n_step, step_size=0.01):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.

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
    loss_arr = np.zeros(n_step).astype("f4")
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)
    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    for istep in range(n_step):
        p = np.array(get_params(opt_state))
        params_arr[istep, :] = p

        msg = "On step {0} some parameters are NaN: {1}".format(istep, p)
        assert np.all(np.isfinite(p)), msg
        loss, grads = value_and_grad(loss_func, argnums=0)(p, loss_data)

        msg = "On step {0} loss is NaN for params:\n{1}".format(istep, p)
        assert np.isfinite(loss), msg

        msg = "On step {0} for params = {1}\n\ngrads = {2}".format(istep, p, grads)
        assert np.all(np.isfinite(grads)), msg
        loss_arr[istep] = loss
        opt_state = opt_update(istep, grads, opt_state)

    good_msk = np.all(np.isfinite(params_arr), axis=1)
    cut_loss_arr = loss_arr[good_msk]
    cut_params_arr = params_arr[good_msk]

    indx_best = np.nanargmin(cut_loss_arr)
    best_fit_params = cut_params_arr[indx_best]
    loss = cut_loss_arr[indx_best]

    return best_fit_params, loss, loss_arr, params_arr
