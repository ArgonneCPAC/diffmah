"""
"""
import numpy as np
from collections import OrderedDict

DEFAULT_PARAMS = OrderedDict(
    logphi1=-2.38, logphi2=-2.82, log10mstar=10.77, alpha1=-0.28, alpha2=-1.48
)


def double_schechter_smf(
    log10m, logphi1=None, logphi2=None, log10mstar=None, alpha1=None, alpha2=None,
):
    """Some docstring.
    """
    if logphi1 is None:
        logphi1 = DEFAULT_PARAMS["logphi1"]
    if logphi2 is None:
        logphi2 = DEFAULT_PARAMS["logphi2"]
    if log10mstar is None:
        log10mstar = DEFAULT_PARAMS["log10mstar"]
    if alpha1 is None:
        alpha1 = DEFAULT_PARAMS["alpha1"]
    if alpha2 is None:
        alpha2 = DEFAULT_PARAMS["alpha2"]

    s1 = _schechter(log10m, logphi1, log10mstar, alpha1)
    s2 = _schechter(log10m, logphi2, log10mstar, alpha2)
    return s1 + s2


def _schechter(log10m, logphi, log10mstar, alpha, m_lower=None):
    A = np.log(10) * 10 ** logphi
    x = log10m - log10mstar
    phi = A * 10 ** (x * (alpha + 1)) * np.exp(-(10 ** x))
    return phi


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
