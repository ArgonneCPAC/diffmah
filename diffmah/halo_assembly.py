"""Models for the mass assembly of dark matter halos.

The individual_halo_assembly_history function implements a model for
dMh(t)/dt and Mh(t) of individual halos.

The mean_halo_mass_assembly_history function implements a model for
<dMh(t)/dt|M0> and <Mh(t)|M0>, the average assembly of halos of present-day mass M0.

"""
from collections import OrderedDict
import numpy as np
from jax import numpy as jax_np
from jax import jit as jax_jit


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
    cosmic_time,
    logm0,
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
        Default is ~13.85 Gyr.

    *mah_params : float, optional
        The slope of dMh(t)/dt is a sigmoid described by 4 parameters: x0, k, ylo, yhi.
        Each of these 4 parameters exhibits power-law scaling with halo mass,
        so that there are 8 total parameters in addition to logm0. They are, in order:

        dmhdt_x0_c0, dmhdt_x0_c1, dmhdt_k_c0, dmhdt_k_c1,
        dmhdt_ylo_c0, dmhdt_ylo_c1, dmhdt_yhi_c0, dmhdt_yhi_c1

        The MEAN_MAH_PARAMS dictionary sets the default value
        of each of these 8 optional keyword arguments.

    Returns
    -------
    logmah : ndarray of shape (n, )
        Base-10 log of halo mass at the input times.
        Halo mass is in units of Msun.

    log_dmhdt : ndarray of shape (n, )
        Base-10 log of halo mass accretion rate at the input times.
        Accretion rate is in units of Msun/yr.

    Notes
    -----
    The returned logmah array has been normalized so that
    its value at the input t0 exactly equals the input logm0.

    """
    logm0, logt, dtarr, indx_t0 = _process_halo_mah_args(logm0, cosmic_time, t0)

    _x = _mean_halo_assembly_jax_kern(
        logt,
        dtarr,
        logm0,
        dmhdt_x0_c0,
        dmhdt_x0_c1,
        dmhdt_k_c0,
        dmhdt_k_c1,
        dmhdt_ylo_c0,
        dmhdt_ylo_c1,
        dmhdt_yhi_c0,
        dmhdt_yhi_c1,
        indx_t0,
    )
    logmah, log_dmhdt = _x
    return np.array(logmah), np.array(log_dmhdt)


def individual_halo_assembly_history(
    cosmic_time,
    logm0,
    t0=TODAY,
    dmhdt_x0=DEFAULT_MAH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
    dmhdt_early_index=DEFAULT_MAH_PARAMS["dmhdt_early_index"],
    dmhdt_late_index=DEFAULT_MAH_PARAMS["dmhdt_late_index"],
):
    """Rolling power-law model for halo mass accretion rate of individual halos.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    dmhdt_x0 : float, optional
        Base-10 log of the time of peak star formation.

    dmhdt_k : float, optional
        Transition speed between early- and late-time power laws indices.

    dmhdt_early_index : float, optional
        Early-time power-law index dMh/dt ~ t**dmhdt_early_index for logt << dmhdt_x0.

    dmhdt_late_index : float, optional
        Late-time power-law index dMh/dt ~ t**dmhdt_late_index for logt >> dmhdt_x0.

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

    Notes
    -----
    The returned logmah array has been normalized so that
    its value at the input t0 exactly equals the input logm0.

    """
    logm0, logt, dtarr, indx_t0 = _process_halo_mah_args(logm0, cosmic_time, t0)

    logmah, log_dmhdt = _individual_halo_assembly_jax_kern(
        logt,
        dtarr,
        logm0,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_t0,
    )
    return np.array(logmah), np.array(log_dmhdt)


@jax_jit
def _individual_halo_assembly_jax_kern(
    logt, dtarr, logm0, dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index, indx_t0
):
    """JAX kernel for the MAH of individual dark matter halos."""
    #  Use a sigmoid to model log10(dMh/dt) with arbitrary normalization
    slope = jax_sigmoid(logt, dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index)
    _log_dmhdt_unnnormed = slope * (logt - logt[indx_t0])

    # Integrate dMh/dt to calculate Mh(t) with arbitrary normalization
    _dmhdt_unnnormed = jax_np.power(10, _log_dmhdt_unnnormed)
    _dmah_unnnormed_integrand = _dmhdt_unnnormed * dtarr
    # in this section could use jax_np.logcumsumexp if it existed
    _mah_unnnormed = jax_np.cumsum(_dmah_unnnormed_integrand) * 1e9
    _logmah_unnnormed = jax_np.log10(_mah_unnnormed)

    # Normalize Mh(t) dMh/dt to integrate to logm0 at logt0
    _logm0_unnnormed = _logmah_unnnormed[indx_t0]
    _rescaling_factor = logm0 - _logm0_unnnormed
    logmah = _logmah_unnnormed + _rescaling_factor
    log_dmhdt = jax_np.log10(_dmhdt_unnnormed) + _rescaling_factor
    return logmah, log_dmhdt


def _mean_halo_assembly_jax_kern(
    logt,
    dtarr,
    logm0,
    dmhdt_x0_c0,
    dmhdt_x0_c1,
    dmhdt_k_c0,
    dmhdt_k_c1,
    dmhdt_ylo_c0,
    dmhdt_ylo_c1,
    dmhdt_yhi_c0,
    dmhdt_yhi_c1,
    indx_t0,
):
    """JAX kernel for the average MAH of halos with present-day mass logm0."""
    dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index = _get_individual_mah_params(
        logm0,
        dmhdt_x0_c0,
        dmhdt_x0_c1,
        dmhdt_k_c0,
        dmhdt_k_c1,
        dmhdt_ylo_c0,
        dmhdt_ylo_c1,
        dmhdt_yhi_c0,
        dmhdt_yhi_c1,
    )

    logmah, log_dmhdt = _individual_halo_assembly_jax_kern(
        logt,
        dtarr,
        logm0,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_t0,
    )
    return logmah, log_dmhdt


def _get_individual_mah_params(
    logm0,
    dmhdt_x0_c0,
    dmhdt_x0_c1,
    dmhdt_k_c0,
    dmhdt_k_c1,
    dmhdt_ylo_c0,
    dmhdt_ylo_c1,
    dmhdt_yhi_c0,
    dmhdt_yhi_c1,
):
    """Calculate the rolling power-law params for the input logm0."""
    x = logm0 - 13
    dmhdt_x0 = dmhdt_x0_c0 + dmhdt_x0_c1 * x
    dmhdt_k = dmhdt_k_c0 + dmhdt_k_c1 * x
    dmhdt_ylo = dmhdt_ylo_c0 + dmhdt_ylo_c1 * x
    dmhdt_yhi = dmhdt_yhi_c0 + dmhdt_yhi_c1 * x
    return dmhdt_x0, dmhdt_k, dmhdt_ylo, dmhdt_yhi


def _process_halo_mah_args(logm0, cosmic_time, t0):
    """Do some bounds-checks and calculate the arrays needed by the MAH kernel."""
    cosmic_time = np.atleast_1d(cosmic_time).astype("f4")
    assert cosmic_time.size > 1, "Input cosmic_time must be an array"

    msg = "Input cosmic_time = {} must be strictly positive and monotonic"
    assert np.all(np.diff(cosmic_time) > 0), msg.format(cosmic_time)
    assert np.all(cosmic_time > 0), msg.format(cosmic_time)

    assert cosmic_time[-1] >= t0 - 0.1, "cosmic_time must span t0"

    logt = np.log10(cosmic_time)
    dtarr = _get_dt_array(cosmic_time)
    present_time_indx = np.argmin(np.abs(cosmic_time - TODAY))

    return logm0, logt, dtarr, present_time_indx


def _get_dt_array(t):
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


def jax_sigmoid(x, x0, k, ylo, yhi):
    return ylo + (yhi - ylo) / (1 + jax_np.exp(-k * (x - x0)))
