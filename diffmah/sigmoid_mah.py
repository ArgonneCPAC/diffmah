"""
"""
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict

__all__ = ("median_logmpeak_from_logt", "logmpeak_from_logt")


DEFAULT_MAH_PARAMS = OrderedDict(
    logtk=3,
    dlogm_height=5,
    logtc_logm0=8.3,
    logtc_k=0.4,
    logtc_ymin=1.635,
    logtc_ymax=-0.365,
    logtc_scatter_dwarfs=0.3,
    logtc_scatter_clusters=0.3,
    logtc_scatter_logmc=12,
    logtc_scatter_logmc_k=1,
)
LOGT0 = 1.14


def median_logmpeak_from_logt(logt, logmpeak_at_t0, logt0=LOGT0, **kwargs):
    """Median halo mass growth vs cosmic time.

    Parameters
    ----------
    logt : float or ndarray
        Base-10 log of cosmic time in Gyr

    logmpeak_at_t0 : float or ndarray
        Base-10 log of peak halo mass in Msun/h

    logt0 : float or ndarray, optional
        Base-10 log of cosmic time in Gyr at the time of halo observation
        Default is log10(13.8) = 1.14

    Returns
    -------
    logmpeak : ndarray
        Base-10 log of halo mass in Msun/h

    """
    logt, logmpeak_at_t0 = _get_1d_arrays(logt, logmpeak_at_t0)
    logtc, logtk, dlogm_height = _median_mah_sigmoid_params(logmpeak_at_t0, **kwargs)
    dlogm = _sigmoid(logt, logtc, logtk, -dlogm_height, 0)
    dlogm_at_logt0 = _sigmoid(logt0, logtc, logtk, -dlogm_height, 0)
    logmpeak = logmpeak_at_t0 + (dlogm - dlogm_at_logt0)
    return logmpeak


def logm0_from_logm_at_logt(
    logt, logm_at_logt, logtc, logtk, dlogm_height, logt0, tol=0.01
):
    """Calculate mass at z=0 from logM(z) and halo MAH params.

    Parameters
    ----------
    logt : float
        Base-10 log of cosmic time where halo attains mass logm_at_logt.

    logm_at_logt : float
        Base-10 log of halo mass at cosmic time logt

    logtc : float
        Base-10 log of the critical time in Gyr.
        Smaller values of logtc produce halos with earlier formation times.

    logtk : float
        Steepness of transition from fast- to slow-accretion regimes.
        Larger values of k produce quicker-transitioning halos.

    dlogm_height : float
        Total gain in logmpeak until logt0

    tol : float, optional
        Numerical tolerance to use in the minimizer. Default is 0.01.

    Returns
    -------
    logm0 : float
        Base-10 log of halo mass at z=0

    """

    def mse_loss(logm0):
        pred_logm_at_logt = logmpeak_from_logt(
            logt, logtc, logtk, dlogm_height, logm0, logt0
        )
        diff = pred_logm_at_logt - logm_at_logt
        return diff * diff

    logm0_guess = (logm_at_logt,)
    res = minimize(mse_loss, logm0_guess, tol=tol)
    return res.x[0]


def logtc_from_logm_at_logt(
    logt, logm_at_logt, logm0_table=np.arange(8, 17, 0.05), **mah_params,
):
    """Calculate logtc parameter for a halo of mass logm_at_logt at logt.

    Parameters
    ----------
    logt : float
        Base-10 log of cosmic time where halo attains mass logm_at_logt.

    logm_at_logt : float or ndarray
        Base-10 log of halo mass at cosmic time logt

    **mah_params : float, optional
        Any parameters in the MAH model dictionary DEFAULT_MAH_PARAMS.

    Returns
    -------
    logtc : float or ndarray
        Parameter of the sigmoid-based model for halo MAH

    """
    logtc_table = _median_mah_sigmoid_params(logm0_table, **mah_params)[0]
    logm_at_logt_table = median_logmpeak_from_logt(logt, logm0_table)
    logtc = np.interp(logm_at_logt, logm_at_logt_table, logtc_table)
    return logtc


def logmpeak_from_logt(logt, logtc, logtk, dlogm_height, logmpeak_at_logt0, logt0):
    """Parametric model for halo mass growth vs cosmic time.

    Parameters
    ----------
    logt : float or ndarray
        Base-10 log of cosmic time in Gyr

    logtc : float or ndarray
        Base-10 log of the critical time in Gyr.
        Smaller values of logtc produce halos with earlier formation times.

    logtk : float or ndarray
        Steepness of transition from fast- to slow-accretion regimes.
        Larger values of k produce quicker-transitioning halos.

    dlogm_height : float or ndarray
        Total gain in logmpeak until logt0

    logmpeak_at_t0 : float or ndarray
        Base-10 log of peak halo mass in Msun/h at the time of halo observation

    logt0 : float or ndarray
        Base-10 log of the time of halo observation.

    Returns
    -------
    logmpeak : ndarray
        Base-10 log of halo mass in Msun/h

    """
    dlogm = _sigmoid(logt, logtc, logtk, -dlogm_height, 0)
    dlogm_at_logt0 = _sigmoid(logt0, logtc, logtk, -dlogm_height, 0)
    logmpeak = logmpeak_at_logt0 + (dlogm - dlogm_at_logt0)
    return logmpeak


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))


def _median_mah_sigmoid_params(logm0, **mah_params):
    """MAH parameters for a halo with present-day mass logm0.
    """
    logtc_logm0 = mah_params.get("logtc_logm0", DEFAULT_MAH_PARAMS["logtc_logm0"])
    logtc_k = mah_params.get("logtc_k", DEFAULT_MAH_PARAMS["logtc_k"])
    logtc_ymin = mah_params.get("logtc_ymin", DEFAULT_MAH_PARAMS["logtc_ymin"])
    logtc_ymax = mah_params.get("logtc_ymax", DEFAULT_MAH_PARAMS["logtc_ymax"])
    logtc = _logtc_param_model(
        logm0,
        logtc_logm0=logtc_logm0,
        logtc_k=logtc_k,
        logtc_ymin=logtc_ymin,
        logtc_ymax=logtc_ymax,
    )

    _logtk = mah_params.get("logtk", DEFAULT_MAH_PARAMS["logtk"])
    logtk = np.zeros_like(logm0) + _logtk

    _dlogm_height = mah_params.get("dlogm_height", DEFAULT_MAH_PARAMS["dlogm_height"])
    dlogm_height = np.zeros_like(logm0) + _dlogm_height

    return logtc, logtk, dlogm_height


def _mah_sigmoid_params_logm_at_logt(
    logt, logm_at_logt, logt0=LOGT0, logtc=None, **mah_params
):
    """
    """
    logt, logm_at_logt = _get_1d_arrays(logt, logm_at_logt)
    n = logm_at_logt.size

    if logtc is None:
        logtc = np.zeros(n)
        for i in range(n):
            logtc[i] = logtc_from_logm_at_logt(logt[i], logm_at_logt[i], **mah_params)
    else:
        logtc = np.zeros(n) + logtc

    logtk = np.zeros(n) + mah_params.get("logtk", DEFAULT_MAH_PARAMS["logtk"])
    dlogm_height = np.zeros(n) + mah_params.get(
        "dlogm_height", DEFAULT_MAH_PARAMS["dlogm_height"]
    )

    logm0 = np.zeros(n)
    for i in range(n):
        logm0[i] = logm0_from_logm_at_logt(
            logt[i], logm_at_logt[i], logtc[i], logtk[i], dlogm_height[i], logt0
        )

    return logtc, logtk, dlogm_height, logm0


def _logtc_from_mah_percentile(logm0, p, **mah_params):
    logtc_scatter_logmc = mah_params.get(
        "logtc_scatter_logmc", DEFAULT_MAH_PARAMS["logtc_scatter_logmc"]
    )
    logtc_scatter_logmc_k = mah_params.get(
        "logtc_scatter_logmc_k", DEFAULT_MAH_PARAMS["logtc_scatter_logmc_k"]
    )
    logtc_scatter_dwarfs = mah_params.get(
        "logtc_scatter_dwarfs", DEFAULT_MAH_PARAMS["logtc_scatter_dwarfs"]
    )
    logtc_scatter_clusters = mah_params.get(
        "logtc_scatter_clusters", DEFAULT_MAH_PARAMS["logtc_scatter_clusters"]
    )
    logtc_scatter = _sigmoid(
        logm0,
        logtc_scatter_logmc,
        logtc_scatter_logmc_k,
        logtc_scatter_dwarfs,
        logtc_scatter_clusters,
    )
    logtc_logm0 = mah_params.get("logtc_logm0", DEFAULT_MAH_PARAMS["logtc_logm0"])
    logtc_k = mah_params.get("logtc_k", DEFAULT_MAH_PARAMS["logtc_k"])
    logtc_ymin = mah_params.get("logtc_ymin", DEFAULT_MAH_PARAMS["logtc_ymin"])
    logtc_ymax = mah_params.get("logtc_ymax", DEFAULT_MAH_PARAMS["logtc_ymax"])
    logtc_med = _logtc_param_model(
        logm0,
        logtc_logm0=logtc_logm0,
        logtc_k=logtc_k,
        logtc_ymin=logtc_ymin,
        logtc_ymax=logtc_ymax,
    )
    logtc_lo, logtc_hi = logtc_med - logtc_scatter, logtc_med + logtc_scatter
    logtc = logtc_lo + p * (logtc_hi - logtc_lo)
    return logtc


def _logtc_param_model(
    logm,
    logtc_logm0=DEFAULT_MAH_PARAMS["logtc_logm0"],
    logtc_k=DEFAULT_MAH_PARAMS["logtc_k"],
    logtc_ymin=DEFAULT_MAH_PARAMS["logtc_ymin"],
    logtc_ymax=DEFAULT_MAH_PARAMS["logtc_ymax"],
):
    return _sigmoid(logm, logtc_logm0, logtc_k, logtc_ymin, logtc_ymax)


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
