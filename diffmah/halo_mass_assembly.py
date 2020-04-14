"""
"""
from jax import numpy as jax_np
from jax import jit as jax_jit
from jax import vmap as jax_vmap
import numpy as np
from collections import OrderedDict

__all__ = ("logmpeak_from_logt",)


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


def _logmpeak_from_logt_jax_kern(logt, params):
    """Parametric model for halo mass growth vs cosmic time.

    Parameters
    ----------
    logt : float or ndarray
        Base-10 log of cosmic time in Gyr

    params : 5-element sequence
        Parameters described in order below

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
    logtc, logtk, dlogm_height, logmpeak_at_logt0, logt0 = params
    dlogm = _jax_sigmoid(logt, logtc, logtk, -dlogm_height, 0)
    dlogm_at_logt0 = _jax_sigmoid(logt0, logtc, logtk, -dlogm_height, 0)
    logmpeak = logmpeak_at_logt0 + (dlogm - dlogm_at_logt0)
    return logmpeak


logmpeak_from_logt = jax_jit(jax_vmap(_logmpeak_from_logt_jax_kern, in_axes=(0, None)))
logmpeak_from_logt.__doc__ = _logmpeak_from_logt_jax_kern.__doc__


def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))
