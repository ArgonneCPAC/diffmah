"""
"""
from collections import OrderedDict
from jax import numpy as jax_np
import numpy as np


DEFAULT_PARAMS = OrderedDict(
    M0=11.339,
    Mz=0.692,
    beta0=3.344,
    betaz=-2.079,
    gamma0=0.966,
    epsilonN0=0.005,
    epsilonNz=0.689,
)


def sfr_efficiency_function(M, z, **kwargs):
    """SFR efficiency function from Moster+17, arXiv:1705.05373.
    See Eqns 5-10.

    Parameters
    ----------
    M : float or ndarray of shape (n, )

    z : float or ndarray of shape (n, )

    params : model parameters, optional

    Returns
    -------
    sfe : ndarray of shape (n, )

    """
    M, z = _get_1d_arrays(M, z)

    param_dict = OrderedDict()
    for param_name, default_value in DEFAULT_PARAMS.items():
        param_dict[param_name] = kwargs.get(param_name, default_value)
    params = param_dict.values()

    return np.array(_sfr_efficiency_function(M, z, params))


def _sfr_efficiency_function(M, z, params):
    M0, Mz, beta0, betaz, gamma0, epsilonN0, epsilonNz = params

    M1 = jax_np.power(10, _get_logM1_param(z, M0, Mz))
    beta = _get_beta_param(z, beta0, betaz)
    gamma = _get_gamma_param(z, gamma0)
    epsilonN = _get_epsilonN_param(z, epsilonN0, epsilonNz)

    return _sf_efficiency_double_powerlaw(M, M1, beta, gamma, epsilonN)


def _sf_efficiency_double_powerlaw(M, M1, beta, gamma, epsilonN):
    x = M / M1
    return 2 * epsilonN / (jax_np.power(x, -beta) + jax_np.power(x, gamma))


def _get_logM1_param(z, M0, Mz):
    return M0 + Mz * z / (1 + z)


def _get_beta_param(z, beta0, betaz):
    return beta0 + betaz * z / (1 + z)


def _get_gamma_param(z, gamma0):
    return gamma0


def _get_epsilonN_param(z, epsilonN0, epsilonNz):
    return epsilonN0 + epsilonNz * z / (1 + z)


def _get_Mmax(M1, beta, gamma):
    return M1 * (beta / gamma) ** (1 / (beta + gamma))


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
