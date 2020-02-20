"""
"""
import numpy as np
from collections import OrderedDict

__all__ = ("median_logmpeak_from_logt", "logmpeak_from_logt")


DEFAULT_MAH_PARAMS = OrderedDict(
    logtk=3,
    dlogm_height=5,
    logtc_logm0=8.3,
    logtc_k=0.4,
    logtc_ymin=1.635,
    logtc_ymax=-0.365,
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


def _median_mah_sigmoid_params(logmpeak, **params):
    logtc_logm0 = params.get("logtc_logm0", DEFAULT_MAH_PARAMS["logtc_logm0"])
    logtc_k = params.get("logtc_k", DEFAULT_MAH_PARAMS["logtc_k"])
    logtc_ymin = params.get("logtc_ymin", DEFAULT_MAH_PARAMS["logtc_ymin"])
    logtc_ymax = params.get("logtc_ymax", DEFAULT_MAH_PARAMS["logtc_ymax"])
    logtc = _logtc_param_model(
        logmpeak,
        logtc_logm0=logtc_logm0,
        logtc_k=logtc_k,
        logtc_ymin=logtc_ymin,
        logtc_ymax=logtc_ymax,
    )

    logtk = np.zeros_like(logmpeak) + DEFAULT_MAH_PARAMS["logtk"]
    dlogm_height = np.zeros_like(logmpeak) + DEFAULT_MAH_PARAMS["dlogm_height"]

    return logtc, logtk, dlogm_height


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
