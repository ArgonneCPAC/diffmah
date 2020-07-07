"""Module implements the individual_log_sfr_history function."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from .main_sequence_sfr_eff import _log_sfr_efficiency_ms_jax_kern
from .halo_assembly import _individual_halo_assembly_jax_kern
from .halo_assembly import _process_halo_mah_args
from .halo_assembly import DEFAULT_MAH_PARAMS, TODAY
from .main_sequence_sfr_eff import DEFAULT_SFR_MS_PARAMS
from .quenching_times import _jax_gradual_quenching


FB = 0.158

QUENCHING_DICT = OrderedDict(log_qtime=0.9, qspeed=5)
DEFAULT_SFRH_PARAMS = OrderedDict()
DEFAULT_SFRH_PARAMS.update(DEFAULT_MAH_PARAMS)
DEFAULT_SFRH_PARAMS.update(DEFAULT_SFR_MS_PARAMS)
DEFAULT_SFRH_PARAMS.update(QUENCHING_DICT)


def individual_sfr_history(
    cosmic_time,
    logmp,
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
    qspeed=DEFAULT_SFRH_PARAMS["qspeed"],
    tmp=TODAY,
):
    """Model for star formation history vs time for a halo with present-day mass logmp.

    Parameters
    ----------
    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    logmp : float
        Base-10 log of peak halo mass in units of Msun

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
        median growth history for a halo of mass logmp.

        See halo_assembly.DEFAULT_MAH_PARAMS for more info on MAH parameters
        See main_sequence_sfr_eff.DEFAULT_SFR_MS_PARAMS
        for more info on SFR efficiency  parameters

    tmp : float, optional
        Age of the universe in Gyr at the time halo mass attains the input logmp.
        There must exist some entry of the input cosmic_time array within 50Myr of tmp.
        Default is ~13.85 Gyr.

    Returns
    -------
    log_sfr : ndarray of shape (n, )
        Base-10 log of star formation rate in units of Msun/yr

    """
    logmp, logt, dtarr, indx_tmp = _process_halo_mah_args(logmp, cosmic_time, tmp)

    log_sfr, log_sm = _individual_log_mstar_history_jax_kern(
        logt,
        dtarr,
        logmp,
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
        qspeed,
        indx_tmp,
    )
    log_sfr, log_sm = np.array(log_sfr), np.array(log_sm)
    return log_sfr, log_sm


def _individual_log_sfr_history_jax_kern(
    logt,
    dtarr,
    logmp,
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
    qspeed,
    indx_tmp,
):

    log_dmhdt = _individual_halo_assembly_jax_kern(
        logt,
        dtarr,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )[1]
    log_dmbdt = jax_np.log10(FB) + log_dmhdt

    log_sfr_eff = _log_sfr_efficiency_ms_jax_kern(
        logt, lge0, k_early, lgtc, lgec, k_trans, a_late
    )
    log_sfr_ms = log_dmbdt + log_sfr_eff
    log_sfr = log_sfr_ms + _jax_gradual_quenching(logt, log_qtime, qspeed)

    return log_sfr


def _calculate_cumulative_in_situ_mass(log_sfr, dtarr):
    log_smh = jax_np.log10(jax_np.cumsum(jax_np.power(10, log_sfr)) * dtarr) + 9.0
    return log_smh


def _individual_log_mstar_history_jax_kern(
    logt,
    dtarr,
    logmp,
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
    qspeed,
    indx_tmp,
):
    log_sfr = _individual_log_sfr_history_jax_kern(
        logt,
        dtarr,
        logmp,
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
        qspeed,
        indx_tmp,
    )
    log_smh = _calculate_cumulative_in_situ_mass(log_sfr, dtarr)
    return log_sfr, log_smh
