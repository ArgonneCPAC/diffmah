"""
"""

import numpy as np
from jax import random as jran
from scipy.stats import random_correlation

from ..utils import (
    _inverse_sigmoid,
    _sigmoid,
    correlation_from_covariance,
    covariance_from_correlation,
    get_cholesky_from_params,
    trimmed_mean,
    trimmed_mean_and_variance,
)


def _enforce_is_cov(matrix):
    det = np.linalg.det(matrix)
    assert det.shape == ()
    assert det > 0
    covinv = np.linalg.inv(matrix)
    assert np.all(np.isfinite(covinv))
    assert np.all(np.isreal(covinv))
    assert np.allclose(matrix, matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    assert np.all(evals > 0)


def test_inverse_sigmoid_actually_inverts():
    """"""
    x0, k, ylo, yhi = 0, 5, 1, 0
    xarr = np.linspace(-1, 1, 100)
    yarr = np.array(_sigmoid(xarr, x0, k, ylo, yhi))
    xarr2 = np.array(_inverse_sigmoid(yarr, x0, k, ylo, yhi))
    assert np.allclose(xarr, xarr2, rtol=1e-3)


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


def test_trimmed_mean_agrees_with_scipy():
    from scipy.stats.mstats import trimmed_mean as trimmed_mean_scipy

    ran_key = jran.key(0)
    ptest = 0.1, 0.2, 0.3
    for p in ptest:
        ran_key, test_key = jran.split(ran_key, 2)
        x = jran.normal(test_key, shape=(20_000,))
        mu_p10 = trimmed_mean(x, p)
        mu_p10_scipy = trimmed_mean_scipy(x, p)
        assert np.allclose(mu_p10, mu_p10_scipy, rtol=0.01)


def test_trimmed_mean_and_variance_agrees_with_scipy():
    from scipy.stats.mstats import trimmed_mean as trimmed_mean_scipy
    from scipy.stats.mstats import trimmed_var as trimmed_var_scipy

    ran_key = jran.key(0)
    ptest = 0.01, 0.1, 0.2, 0.3
    for p in ptest:
        ran_key, test_key = jran.split(ran_key, 2)
        x = jran.normal(test_key, shape=(20_000,))
        mu_p10, var_p10 = trimmed_mean_and_variance(x, p)
        mu_p10_scipy = trimmed_mean_scipy(x, p)
        var_p10_scipy = trimmed_var_scipy(x, p)
        assert np.allclose(mu_p10, mu_p10_scipy, rtol=0.01)
        assert np.allclose(var_p10, var_p10_scipy, rtol=0.01)


def test_trimmed_mean_and_variance_consistency():
    ran_key = jran.key(0)
    x = jran.normal(ran_key, shape=(20_000,))
    mu, var = trimmed_mean_and_variance(x, 0.1)
    mu2 = trimmed_mean(x, 0.1)
    assert np.allclose(mu, mu2, rtol=1e-4)


def test_correlation_from_covariance():
    ntests = 100
    for __ in range(ntests):
        ndim = np.random.randint(2, 10)
        evals = np.sort(np.random.uniform(0, 100, ndim))
        evals = ndim * evals / evals.sum()
        corr_matrix = random_correlation.rvs(evals)
        cov_matrix = covariance_from_correlation(corr_matrix, evals)
        S = np.sqrt(np.diag(cov_matrix))
        assert np.allclose(S, evals, rtol=1e-4)
        inferred_corr_matrix = correlation_from_covariance(cov_matrix)
        assert np.allclose(corr_matrix, inferred_corr_matrix, rtol=1e-4)
        _enforce_is_cov(cov_matrix)
