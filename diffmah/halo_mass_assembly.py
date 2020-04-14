"""
"""
import numpy as np
from jax import numpy as jax_np
from jax import jit as jax_jit
from jax import vmap as jax_vmap
from jax import grad as jax_grad
from collections import OrderedDict

__all__ = ("halo_mass_vs_time",)


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
COSMIC_AGE_TODAY = 10 ** LOGT0


def halo_mass_vs_time(
    cosmic_time, logm0, t0=COSMIC_AGE_TODAY, mah_percentile=None, **kwargs
):
    """Median halo mass growth vs cosmic time.

    Parameters
    ----------
    cosmic_time : float or ndarray
        Age of the Universe in Gyr

    logm0 : float or ndarray
        Base-10 log of peak halo mass in Msun/h

    t0 : float, optional
        Present-day age of the Universe in Gyr. Default is 13.8.

    Returns
    -------
    logmh : ndarray
        Base-10 log of halo mass in Msun/h

    """
    cosmic_time, logm0 = _get_1d_arrays(cosmic_time, logm0, dtype="f4")
    logtc, logtk, dlogm_height = _get_mah_sigmoid_params(
        logm0, mah_percentile=mah_percentile, **kwargs
    )
    params = logtc, logtk, dlogm_height, logm0
    logmh = np.array(logmpeak_vs_time_jax(cosmic_time, t0, params))
    return logmh


def _logmpeak_vs_time_jax_kern(cosmic_time, t0, params):
    """Parametric model for halo mass growth vs cosmic time.

    Parameters
    ----------
    cosmic_time : float or ndarray
        Age of the Universe in Gyr

    t0 : float
        Present-day age of the Universe in Gyr

    params : 4-element sequence
        Parameters described in order below

    logtc : float or ndarray
        Base-10 log of the critical time in Gyr.
        Smaller values of logtc produce halos with earlier formation times.

    logtk : float or ndarray
        Steepness of transition from fast- to slow-accretion regimes.
        Larger values of k produce quicker-transitioning halos.

    dlogm_height : float or ndarray
        Total gain in logmpeak until logt0

    logm0 : float or ndarray
        Base-10 log of halo mass at t0

    Returns
    -------
    logmpeak : ndarray
        Base-10 log of halo mass

    """
    logtc, logtk, dlogm_height, logm0 = params
    logt = jax_np.log10(cosmic_time)
    logt0 = jax_np.log10(t0)

    dlogm = _jax_sigmoid(logt, logtc, logtk, -dlogm_height, 0)
    dlogm_at_logt0 = _jax_sigmoid(logt0, logtc, logtk, -dlogm_height, 0)
    logmpeak = logm0 + (dlogm - dlogm_at_logt0)
    return logmpeak


logmpeak_vs_time_jax = jax_jit(
    jax_vmap(_logmpeak_vs_time_jax_kern, in_axes=(0, None, 0))
)
logmpeak_vs_time_jax.__doc__ = _logmpeak_vs_time_jax_kern.__doc__


def _halo_dmhdt_integral(cosmic_time, t0, params):
    return jax_np.power(10.0, _logmpeak_vs_time_jax_kern(cosmic_time, t0, params))


halo_dmdt_vs_time_jax_kern = jax_grad(_halo_dmhdt_integral, argnums=0)
halo_dmdt_vs_time_jax = jax_jit(
    jax_vmap(halo_dmdt_vs_time_jax_kern, in_axes=(0, None, 0))
)


def halo_dmdt_vs_time(
    cosmic_time, logm0, t0=COSMIC_AGE_TODAY, mah_percentile=None, **kwargs
):
    """Median halo mass accretion rate vs cosmic time.

    Parameters
    ----------
    cosmic_time : float or ndarray
        Age of the Universe in Gyr

    logm0 : float or ndarray
        Base-10 log of peak halo mass in Msun/h

    t0 : float, optional
        Present-day age of the Universe in Gyr. Default is 13.8.

    Returns
    -------
    dmhdt : ndarray
        dMhalo / dt in units of Msun/yr

    """
    cosmic_time, logm0 = _get_1d_arrays(cosmic_time, logm0, dtype="f4")
    logtc, logtk, dlogm_height = _get_mah_sigmoid_params(
        logm0, mah_percentile=mah_percentile, **kwargs
    )
    params = logtc, logtk, dlogm_height, logm0
    dmhdt = np.array(halo_dmdt_vs_time_jax(cosmic_time, t0, params)) / 1e9
    return dmhdt


def _get_mah_sigmoid_params(logm0, mah_percentile=None, **kwargs):
    """MAH parameters for a halo with present-day mass logm0.
    """
    logm0 = np.atleast_1d(logm0).astype("f4")

    logtc_keys = ("logtc_logm0", "logtc_k", "logtc_ymin", "logtc_ymax")
    if mah_percentile is None:
        logtc_params = OrderedDict(
            [(key, kwargs.get(key, DEFAULT_MAH_PARAMS[key])) for key in logtc_keys]
        )
        logtc_med = np.zeros_like(logm0) + _logtc_param_model(logm0, **logtc_params)
        logtc = np.zeros_like(logm0) + kwargs.get("logtc", logtc_med)
    else:
        msg = "Do not pass mah_percentile argument and {0} argument"
        for key in logtc_keys:
            assert key not in kwargs, msg.format(key)
        logtc = np.zeros_like(logm0) + _logtc_from_mah_percentile(
            logm0, mah_percentile, **kwargs
        )

    _logtk = kwargs.get("logtk", DEFAULT_MAH_PARAMS["logtk"])
    logtk = np.zeros_like(logm0) + _logtk

    _dlogm_height = kwargs.get("dlogm_height", DEFAULT_MAH_PARAMS["dlogm_height"])
    dlogm_height = np.zeros_like(logm0) + _dlogm_height

    return logtc, logtk, dlogm_height


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
    logtc_scatter = _jax_sigmoid(
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
    return _jax_sigmoid(logm, logtc_logm0, logtc_k, logtc_ymin, logtc_ymax)


def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))


def _get_1d_arrays(*args, dtype=None):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    if dtype is None:
        return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    else:
        return [np.zeros(npts).astype(dtype) + arr for arr in results]
