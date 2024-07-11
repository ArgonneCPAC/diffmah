"""Utility functions used throughout the package."""

import numpy as np
from jax import jit as jjit
from jax import nn
from jax import numpy as jnp
from jax.example_libraries import optimizers as jax_opt


def get_1d_arrays(*args, jax_arrays=False):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [jnp.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)

    if jax_arrays:
        result = [jnp.zeros(npts).astype(arr.dtype) + arr for arr in results]
    else:
        result = [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    return result


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
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
    return ylo + (yhi - ylo) * nn.sigmoid(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
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
    return x0 - jnp.log(lnarg) / k


@jjit
def _sig_slope(x, xtp, ytp, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return ytp + slope * (x - xtp)


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


def get_dt_array(t):
    """Compute delta time from input time.

    Parameters
    ----------
    t : ndarray of shape (n, )

    Returns
    -------
    dt : ndarray of shape (n, )

    Returned dt is defined by time interval (t_lo, t_hi),
    where t_lo^i = 0.5(t_i-1 + t_i) and t_hi^i = 0.5(t_i + t_i+1)

    """
    n = t.size
    dt = np.zeros(n)
    tlo = t[0] - (t[1] - t[0]) / 2
    for i in range(n - 1):
        thi = (t[i + 1] + t[i]) / 2
        dt[i] = thi - tlo
        tlo = thi
    thi = t[n - 1] + dt[n - 2] / 2
    dt[n - 1] = thi - tlo
    return dt


@jjit
def get_cholesky_from_params(params):
    """Compute the Cholesky matrix defined by a parameter array

    Parameters
    ----------
    params : ndarray of shape (n_params, )

    Returns
    -------
    cholesky : ndarray of shape (ndim, ndim)
        Lower triangular matrix with ndim = 0.5*nparams*(nparams+1)
        The first ndim entries of params are stored along the diagonal
        The remaining entries of params are stored as the off-diagonal entries
        according to the following convention

        params = [1, 2, 3, 4, 5, 6]
        cholesky = Array([[1, 0, 0], [4, 2, 0], [5, 6, 3]])

    """
    ndim = ((np.sqrt(8 * params.size + 1) - 1) / 2).astype(int)
    indx_diags = np.diag(np.arange(ndim).astype(int))
    unity = jnp.tril(jnp.ones((ndim, ndim)))
    lunity = jnp.abs(unity * (jnp.eye(ndim) - 1)).astype(int)
    X = (jnp.cumsum(lunity).reshape((ndim, ndim)) * lunity + ndim) * lunity
    indx_offdiags = (X - 1) * lunity
    indx = indx_diags + indx_offdiags
    cholesky = params[indx.flatten()].reshape((ndim, ndim)) * unity
    return cholesky


@jjit
def trimmed_mean(x, p_trim):
    """Mean of x after excluding points outside p_trim percentile
    Equivalent to scipy.stats.mstats.trimmed_mean

    Parameters
    ----------
    x : array

    p_trim : float
        0 <= p < 0.5

    Returns
    -------
    mu : float

    """
    xlo = jnp.percentile(x, p_trim * 100.0)
    xhi = jnp.percentile(x, (1.0 - p_trim) * 100.0)
    xmsk = (x > xlo) & (x < xhi)
    _weights = jnp.where(xmsk, 1.0, 0.0)
    weights = _weights / _weights.sum()
    trimmed_mean = jnp.sum(x * weights)
    return trimmed_mean


@jjit
def trimmed_mean_and_variance(x, p_trim):
    """Mean and variance of x after excluding points outside p_trim percentile.
    Equivalent to scipy.stats.mstats.trimmed_mean and scipy.stats.mstats.trimmed_var

    Parameters
    ----------
    x : array

    p_trim : float
        0 <= p < 0.5

    Returns
    -------
    mu : float

    var : float

    """
    xlo = jnp.percentile(x, p_trim * 100.0)
    xhi = jnp.percentile(x, (1.0 - p_trim) * 100.0)
    xmsk = (x > xlo) & (x < xhi)
    _weights = jnp.where(xmsk, 1.0, 0.0)
    weights = _weights / _weights.sum()

    trimmed_mean = jnp.sum(x * weights)

    num_nonzero_weights = jnp.sum(weights > 0.0)
    var_integrand = weights * (x - trimmed_mean) ** 2
    numerator = jnp.sum(var_integrand)
    denominator = (num_nonzero_weights - 1) / num_nonzero_weights
    trimmed_variance = numerator / denominator

    return trimmed_mean, trimmed_variance


@jjit
def covariance_from_correlation(corr, evals):
    """Covariance matrix from correlation matrix

    Parameters
    ----------
    corr : array, shape (n, n)

    evals : array, shape (n, )
        Array of eigenvalues, e.g. (σ_1, σ_2, ..., σ_n)
        Note that np.diag(cov) = evals**2

    Returns
    -------
    cov : array, shape (n, n)

    """
    D = jnp.diag(evals)
    cov = jnp.dot(jnp.dot(D, corr), D)
    return cov


@jjit
def correlation_from_covariance(cov):
    """Correlation matrix from covariance matrix

    Parameters
    ----------
    cov : array, shape (n, n)

    Returns
    -------
    corr : array, shape (n, n)

    """
    v = jnp.sqrt(jnp.diag(cov))
    outer_v = jnp.outer(v, v)
    corr = cov / outer_v
    msk = cov == 0
    corr = jnp.where(msk, 0.0, corr)
    return corr
