"""Utility functions used throughout the package."""

import numpy as np
from jax import jit as jjit
from jax import nn
from jax import numpy as jnp


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
