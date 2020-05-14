"""
"""
from collections import OrderedDict
from jax import numpy as jax_np
from jax import jit as jax_jit
from jax import vmap as jax_vmap
from jax.ops import index_update as jax_index_update
from jax.ops import index as jax_index
from .utils import jax_sigmoid


__all__ = ("mean_halo_mass_assembly_history", "individual_halo_assembly_history")

DEFAULT_MAH_PARAMS = OrderedDict(
    dmhdt_x0=0.15, dmhdt_k=4.2, dmhdt_early_index=0.45, dmhdt_late_index=-1.25
)
MEAN_MAH_PARAMS = OrderedDict(
    dmhdt_x0_c0=0.27,
    dmhdt_x0_c1=0.12,
    dmhdt_k_c0=4.7,
    dmhdt_k_c1=0.5,
    dmhdt_ylo_c0=0.7,
    dmhdt_ylo_c1=0.25,
    dmhdt_yhi_c0=-1.0,
    dmhdt_yhi_c1=0.25,
)
LOGT0 = 1.14
TODAY = 10.0 ** LOGT0


def mean_halo_mass_assembly_history(
    logm0,
    cosmic_time,
    t0=TODAY,
    dmhdt_x0_c0=MEAN_MAH_PARAMS["dmhdt_x0_c0"],
    dmhdt_x0_c1=MEAN_MAH_PARAMS["dmhdt_x0_c1"],
    dmhdt_k_c0=MEAN_MAH_PARAMS["dmhdt_k_c0"],
    dmhdt_k_c1=MEAN_MAH_PARAMS["dmhdt_k_c1"],
    dmhdt_ylo_c0=MEAN_MAH_PARAMS["dmhdt_ylo_c0"],
    dmhdt_ylo_c1=MEAN_MAH_PARAMS["dmhdt_ylo_c1"],
    dmhdt_yhi_c0=MEAN_MAH_PARAMS["dmhdt_yhi_c0"],
    dmhdt_yhi_c1=MEAN_MAH_PARAMS["dmhdt_yhi_c1"],
):
    """Rolling power-law model for halo mass accretion rate
    averaged over host halos with present-day mass logm0.

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
    logm0, tarr, logt0, indx_t0 = _process_halo_mah_args(logm0, cosmic_time, t0)

    mean_mah_params = (
        dmhdt_x0_c0,
        dmhdt_x0_c1,
        dmhdt_k_c0,
        dmhdt_k_c1,
        dmhdt_ylo_c0,
        dmhdt_ylo_c1,
        dmhdt_yhi_c0,
        dmhdt_yhi_c1,
    )

    logmah, log_dmhdt = _mean_halo_assembly(
        mean_mah_params, tarr, logm0, indx_t0, logt0
    )

    return logmah, log_dmhdt


def individual_halo_assembly_history(
    logm0,
    cosmic_time,
    dmhdt_x0=DEFAULT_MAH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
    dmhdt_early_index=DEFAULT_MAH_PARAMS["dmhdt_early_index"],
    dmhdt_late_index=DEFAULT_MAH_PARAMS["dmhdt_late_index"],
    t0=TODAY,
):
    """Rolling power-law model for halo mass accretion rate.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    dmhdt_x0 : float, optional
        Base-10 log of peak time of star formation.
        Default is set according to mean value for logm0.

    dmhdt_k : float, optional
        Transition speed between early- and late-time power laws.
        Default is set according to mean value for logm0.

    dmhdt_early_index : float, optional
        Early-time power-law index dMh/dt ~ t**dmhdt_early_index for logt << dmhdt_x0.
        Default is set according to mean value for logm0.

    dmhdt_late_index : float, optional
        Late-time power-law index dMh/dt ~ t**dmhdt_late_index for logt >> dmhdt_x0.
        Default is set according to mean value for logm0.

    t0 : float, optional
        Age of the universe in Gyr at the time halo mass attains the input logm0.
        There must exist some entry of the input cosmic_time array within 50Myr of t0.
        Default is ~13.85 Gyr.

    Returns
    -------
    logmah : ndarray of shape (n, )
        Base-10 log of halo mass at the input times.
        Halo mass is in units of Msun.

    log_dmhdt : ndarray of shape (n, )
        Base-10 log of halo mass accretion rate at the input times
        Accretion rate is in units of Msun/yr.
        By construction, the two returned arrays are mutually consistent
        in that the time integral of log_dmhdt equals logmah.

    Notes
    -----
    The returned logmah array has been normalized so that
    its value at the input t0 exactly equals the input logm0.

    """
    _x = _process_halo_mah_args(logm0, cosmic_time, t0)
    logm0, tarr, logt0, indx_t0 = _x

    mah_params = dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index

    logmah, log_dmhdt = _individual_halo_assembly(
        mah_params, tarr, logm0, indx_t0, logt0
    )
    return logmah, log_dmhdt


def _individual_halo_assembly(individual_mah_params, tarr, logm0, indx_t0, logt0):
    """
    """
    n = tarr.size
    _dt0 = tarr[1] - tarr[0]
    _half_step = (tarr[2] - tarr[1]) / 2  # assumes linear spacing of tarr
    _t_init = max(0.5 * tarr[0], tarr[0] - _dt0)
    _tarr = jax_index_update(jax_np.zeros(n + 1) + _t_init, jax_index[1:], tarr)

    logt0 = jax_np.log10(tarr[indx_t0])
    ti, tf = _tarr[:-1] + _half_step, _tarr[1:] + _half_step
    _dmah, _dmhdt_at_tmid = _jax_normed_dmah(individual_mah_params, ti, tf, logt0)
    _logmah = jax_np.log10(jax_np.cumsum(_dmah)) + 9

    #  Normalize Mh(t) and dMh/dt to integrate to logm0 at logt0
    _logm0 = _logmah[indx_t0]
    rescaling_factor = logm0 - _logm0
    logmah = _logmah + rescaling_factor
    log_dmhdt = jax_np.log10(_dmhdt_at_tmid) + rescaling_factor
    return logmah, log_dmhdt


def _mean_halo_assembly(mean_mah_params, tarr, logm0, indx_t0, logt0):
    individual_mah_params = _get_individual_mah_params(mean_mah_params, logm0)
    return _individual_halo_assembly(individual_mah_params, tarr, logm0, indx_t0, logt0)


def _jax_normed_halo_dmdt_vs_time_kern(mah_params, logt, logt0):
    x0, k, ylo, yhi = mah_params
    return jax_sigmoid(logt, x0, k, ylo, yhi) * (logt - logt0)


def _jax_normed_dmah_kern(mah_params, ti, tf, logt0):
    logtmid = jax_np.log10(0.5 * (ti + tf))
    _dlogmh_dt_at_tmid = _jax_normed_halo_dmdt_vs_time_kern(mah_params, logtmid, logt0)
    _dmhdt_at_tmid = jax_np.power(10, _dlogmh_dt_at_tmid)
    _dmah = _dmhdt_at_tmid * (tf - ti)  # midpoint rule
    return _dmah, _dmhdt_at_tmid


_jax_normed_dmah = jax_jit(jax_vmap(_jax_normed_dmah_kern, in_axes=(None, 0, 0, None)))


def _get_individual_mah_params(mean_mah_params, logm0):
    dmhdt_x0_c0, dmhdt_x0_c1 = mean_mah_params[0:2]
    dmhdt_k_c0, dmhdt_k_c1 = mean_mah_params[2:4]
    dmhdt_ylo_c0, dmhdt_ylo_c1 = mean_mah_params[4:6]
    dmhdt_yhi_c0, dmhdt_yhi_c1 = mean_mah_params[6:8]

    dmhdt_x0 = _get_dmhdt_param(logm0, dmhdt_x0_c0, dmhdt_x0_c1)
    dmhdt_k = _get_dmhdt_param(logm0, dmhdt_k_c0, dmhdt_k_c1)
    dmhdt_ylo = _get_dmhdt_param(logm0, dmhdt_ylo_c0, dmhdt_ylo_c1)
    dmhdt_yhi = _get_dmhdt_param(logm0, dmhdt_yhi_c0, dmhdt_yhi_c1)

    return dmhdt_x0, dmhdt_k, dmhdt_ylo, dmhdt_yhi


def _get_dmhdt_param(logm0, c0, c1):
    return c0 + c1 * (logm0 - 13)


def _process_halo_mah_args(logm0, cosmic_time, t0):
    cosmic_time = jax_np.atleast_1d(cosmic_time).astype("f4")
    assert cosmic_time.size > 1, "Input cosmic_time must be an array"

    msg = "Input cosmic_time = {} must be strictly monotonic"
    _dtarr = jax_np.diff(cosmic_time)
    assert jax_np.all(_dtarr > 0), msg.format(cosmic_time)

    msg = "cosmic_time must be linearly spaced"
    assert jax_np.allclose(_dtarr, _dtarr.mean(), atol=0.05), msg

    dt = jax_np.mean(_dtarr)
    present_time_indx = int(jax_np.round((t0 - cosmic_time[0]) / dt))
    msg = "t0 must lie in the range spanned by cosmic_time"
    assert 0 <= present_time_indx <= cosmic_time.size - 1, msg

    implied_t0 = cosmic_time[present_time_indx]
    logt0 = jax_np.log10(implied_t0)

    return logm0, cosmic_time, logt0, present_time_indx
