"""JAX implementation of double-power-law model of halo mass accretion rate."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from jax import jit as jax_jit
from jax import vmap as jax_vmap
from .utils import jax_sigmoid

__all__ = ("halo_mass_assembly_history",)


DEFAULT_MAH_PARAMS = OrderedDict(
    dmhdt_x0=0.18, dmhdt_k=4.15, dmhdt_early_index=0.4, dmhdt_late_index=-1.25
)

LOGT0 = 1.14
TODAY = 10.0 ** LOGT0


def halo_mass_assembly_history(logm0, cosmic_time, t0=TODAY, **kwargs):
    """Rolling power-law model for halo mass accretion rate.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    t0 : float, optional
        Age of the universe in Gyr at the time halo mass attains the input logm0.
        There must exist some entry of the input cosmic_time array within 50Myr of t0.
        Default is ~13.8 Gyr.

    *mah_params : float, optional
        Any parameter in DEFAULT_MAH_PARAMS is an acceptable keyword argument.

    Returns
    -------
    logmah : ndarray of shape (n, )
        Base-10 log of halo mass at the input times.
        Halo mass is in units of Msun.

    log_dmhdt : ndarray of shape (n, )
        Base-10 log of halo mass accretion rate at the input times
        Accretion rate is in units of Msun/yr.
        By construction, the time integral of log_dmhdt equals logmah.

    Notes
    -----
    The logmah array has been normalized so that its value at t0 exactly equals logm0.

    """
    _x = _process_args(logm0, cosmic_time, t0, **kwargs)
    logt0, _tarr, present_time_indx, params = _x

    logmah, log_dmhdt = _halo_mass_assembly_function(
        params, _tarr, logm0, present_time_indx, logt0
    )
    return logmah, log_dmhdt


def _halo_mass_assembly_function(params, tarr, logm0, present_time_indx, logt0):
    """
    """
    _dmah, _dmhdt_at_tmid = _jax_normed_dmah(params, tarr[:-1], tarr[1:], logt0)
    _logmah = jax_np.log10(jax_np.cumsum(_dmah)) + 9

    #  Normalize Mh(t) and dMh/dt to integrate to logm0 at logt0
    _logm0 = _logmah[present_time_indx]
    rescaling_factor = logm0 - _logm0
    logmah = _logmah + rescaling_factor
    log_dmhdt = jax_np.log10(_dmhdt_at_tmid) + rescaling_factor
    return logmah, log_dmhdt


def _jax_normed_halo_dmdt_vs_time_kern(logt, params, logt0):
    x0, k, ylo, yhi = params
    return jax_sigmoid(logt, x0, k, ylo, yhi) * (logt - logt0)


def _jax_normed_dmah_kern(params, ti, tf, logt0):
    logtmid = jax_np.log10(0.5 * (ti + tf))
    _dlogmh_dt_at_tmid = _jax_normed_halo_dmdt_vs_time_kern(logtmid, params, logt0)
    _dmhdt_at_tmid = jax_np.power(10, _dlogmh_dt_at_tmid)
    _dmah = _dmhdt_at_tmid * (tf - ti)  # midpoint rule
    return _dmah, _dmhdt_at_tmid


_jax_normed_dmah = jax_jit(jax_vmap(_jax_normed_dmah_kern, in_axes=(None, 0, 0, None)))


def _process_args(logm0, cosmic_time, t0, **kwargs):
    """
    """
    # Enforce no unexpected keywords
    expected_kwargs = list(DEFAULT_MAH_PARAMS.keys())
    try:
        assert set(kwargs) <= set(expected_kwargs)
    except AssertionError:
        msg = "Unexpected keyword `{}` passed to halo_mass_assembly_history function"
        unexpected_kwarg = list(set(kwargs) - set(expected_kwargs))[0]
        raise KeyError(msg.format(unexpected_kwarg))

    # Bounds-check input cosmic_time
    cosmic_time = np.atleast_1d(cosmic_time)
    assert cosmic_time.size > 1, "Input cosmic_time must be an array"

    msg = "Input cosmic_time = {} must be strictly monotonic"
    assert np.all(np.diff(cosmic_time) > 0), msg.format(cosmic_time)

    present_time_indx = min(np.searchsorted(cosmic_time, t0), cosmic_time.size - 1)
    implied_t0 = cosmic_time[present_time_indx]
    msg = "Input cosmic_time must have an entry within 50 Myr of t0 = {}"
    assert np.allclose(t0, implied_t0, atol=0.05), msg.format(t0)

    t_init = cosmic_time[0]
    dt_init = cosmic_time[1] - t_init
    new_t_init = max(t_init - dt_init, 0.9 * t_init)
    _tarr = np.insert(cosmic_time, 0, new_t_init)

    # Retrieve MAH params
    params = tuple((kwargs.get(key, val) for key, val in DEFAULT_MAH_PARAMS.items()))

    logt0 = np.log10(t0)
    return logt0, _tarr, present_time_indx, params
