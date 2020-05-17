"""Module implements the individual_log_sfr_history function."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from .main_sequence_sfr_eff import _log_sfr_efficiency_ms_jax_kern
from .halo_assembly import _individual_halo_assembly_jax_kern
from .halo_assembly import _process_halo_mah_args
from .halo_assembly import DEFAULT_MAH_PARAMS, TODAY
from .main_sequence_sfr_eff import DEFAULT_SFR_MS_PARAMS
from .quenching_times import _jax_log_quenching_function


FB = 0.158

QUENCHING_DICT = OrderedDict(log_qtime=1.17, log_qfrac=-2)
DEFAULT_SFRH_PARAMS = OrderedDict()
DEFAULT_SFRH_PARAMS.update(DEFAULT_MAH_PARAMS)
DEFAULT_SFRH_PARAMS.update(DEFAULT_SFR_MS_PARAMS)
DEFAULT_SFRH_PARAMS.update(QUENCHING_DICT)


def individual_sfr_history(
    logm0,
    cosmic_time,
    dmhdt_x0=DEFAULT_SFRH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_SFRH_PARAMS["dmhdt_k"],
    dmhdt_early_index=DEFAULT_SFRH_PARAMS["dmhdt_early_index"],
    dmhdt_late_index=DEFAULT_SFRH_PARAMS["dmhdt_late_index"],
    lge0=DEFAULT_SFRH_PARAMS["lge0"],
    k_early=DEFAULT_SFRH_PARAMS["k_early"],
    lgtc=DEFAULT_SFRH_PARAMS["lgtc"],
    lgec=DEFAULT_SFRH_PARAMS["lgec"],
    k_trans=DEFAULT_SFRH_PARAMS["k_trans"],
    a_late=DEFAULT_SFRH_PARAMS["a_late"],
    log_qtime=DEFAULT_SFRH_PARAMS["log_qtime"],
    log_qfrac=DEFAULT_SFRH_PARAMS["log_qfrac"],
    t0=TODAY,
):
    """Model for star formation history vs time for a halo with present-day mass logm0.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    qtime : float, optional
        Quenching time in units of Gyr.
        Default is 14 for negligible quenching before z=0.

    lge0 : float, optional
        Asymptotic value of SFR efficiency at early times.
        Default set according to average value for Milky Way halos.

    lgtc : float, optional
        Time of peak star formation in Gyr.
        Default set according to average value for Milky Way halos.

    lgec : float, optional
        Normalization of SFR efficiency at the time of peak SFR.
        Default set according to average value for Milky Way halos.

    a_late : float, optional
        Late-time power-law index of SFR efficiency.
        Default set according to average value for Milky Way halos.

    Additional MAH parameters:
            dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index

        Unspecified MAH parameters will be set according to the
        median growth history for a halo of mass logm0.

        See halo_assembly.DEFAULT_MAH_PARAMS for more info on MAH parameters
        See main_sequence_sfr_eff.DEFAULT_SFR_MS_PARAMS
        for more info on SFR efficiency  parameters

    t0 : float, optional
        Age of the universe in Gyr at the time halo mass attains the input logm0.
        There must exist some entry of the input cosmic_time array within 50Myr of t0.
        Default is ~13.85 Gyr.

    Returns
    -------
    log_sfr : ndarray of shape (n, )
        Base-10 log of star formation rate in units of Msun/yr

    """
    logm0, logt, dtarr, indx_t0 = _process_halo_mah_args(logm0, cosmic_time, t0)

    log_sfr, log_sm = _individual_log_mstar_history_jax_kern(
        logm0,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        lge0,
        k_early,
        lgtc,
        lgec,
        k_trans,
        a_late,
        log_qtime,
        log_qfrac,
        logt,
        dtarr,
        indx_t0,
    )
    log_sfr, log_sm = np.array(log_sfr), np.array(log_sm)
    return log_sfr, log_sm


def _individual_log_sfr_history_jax_kern(
    logm0,
    dmhdt_x0,
    dmhdt_k,
    dmhdt_early_index,
    dmhdt_late_index,
    lge0,
    k_early,
    lgtc,
    lgec,
    k_trans,
    a_late,
    log_qtime,
    log_qfrac,
    logt,
    dtarr,
    indx_t0,
):

    log_dmhdt = _individual_halo_assembly_jax_kern(
        logm0,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        logt,
        dtarr,
        indx_t0,
    )[1]
    log_dmbdt = jax_np.log10(FB) + log_dmhdt

    log_sfr_eff = _log_sfr_efficiency_ms_jax_kern(
        logt, lge0, k_early, lgtc, lgec, k_trans, a_late
    )
    log_sfr_ms = log_dmbdt + log_sfr_eff
    log_sfr = log_sfr_ms + _jax_log_quenching_function(logt, log_qtime, log_qfrac)
    return log_sfr


def _calculate_cumulative_in_situ_mass(log_sfr, dtarr):
    log_smh = jax_np.log10(jax_np.cumsum(jax_np.power(10, log_sfr)) * dtarr) + 9.0
    return log_smh


def _individual_log_mstar_history_jax_kern(
    logm0,
    dmhdt_x0,
    dmhdt_k,
    dmhdt_early_index,
    dmhdt_late_index,
    lge0,
    k_early,
    lgtc,
    lgec,
    k_trans,
    a_late,
    log_qtime,
    log_qfrac,
    logt,
    dtarr,
    indx_t0,
):
    log_sfr = _individual_log_sfr_history_jax_kern(
        logm0,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        lge0,
        k_early,
        lgtc,
        lgec,
        k_trans,
        a_late,
        log_qtime,
        log_qfrac,
        logt,
        dtarr,
        indx_t0,
    )
    log_smh = _calculate_cumulative_in_situ_mass(log_sfr, dtarr)
    return log_sfr, log_smh
