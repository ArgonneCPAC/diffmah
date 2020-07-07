"""Module implementing the mean_sfr_history function for the SFR history of
central galaxies averaged over halos of the same present-day mass."""
import numpy as np
from jax import numpy as jax_np
from .halo_assembly import _mean_halo_assembly_jax_kern
from .main_sequence_sfr_eff import mean_log_sfr_efficiency_ms_jax
from .quenching_history import _mean_log_main_sequence_fraction
from .halo_assembly import _process_halo_mah_args
from .halo_assembly import MEAN_MAH_PARAMS, TODAY
from .main_sequence_sfr_eff import MEAN_SFR_MS_PARAMS
from .quenching_history import MEAN_Q_PARAMS

FB = 0.158


def mean_sfr_history(
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
    lge0_lgmc=MEAN_SFR_MS_PARAMS["lge0_lgmc"],
    lge0_at_lgmc=MEAN_SFR_MS_PARAMS["lge0_at_lgmc"],
    lge0_early_slope=MEAN_SFR_MS_PARAMS["lge0_early_slope"],
    lge0_late_slope=MEAN_SFR_MS_PARAMS["lge0_late_slope"],
    k_early_x0=MEAN_SFR_MS_PARAMS["k_early_x0"],
    k_early_k=MEAN_SFR_MS_PARAMS["k_early_k"],
    k_early_ylo=MEAN_SFR_MS_PARAMS["k_early_ylo"],
    k_early_yhi=MEAN_SFR_MS_PARAMS["k_early_yhi"],
    lgtc_x0=MEAN_SFR_MS_PARAMS["lgtc_x0"],
    lgtc_k=MEAN_SFR_MS_PARAMS["lgtc_k"],
    lgtc_ylo=MEAN_SFR_MS_PARAMS["lgtc_ylo"],
    lgtc_yhi=MEAN_SFR_MS_PARAMS["lgtc_yhi"],
    lgec_x0=MEAN_SFR_MS_PARAMS["lgec_x0"],
    lgec_k=MEAN_SFR_MS_PARAMS["lgec_k"],
    lgec_ylo=MEAN_SFR_MS_PARAMS["lgec_ylo"],
    lgec_yhi=MEAN_SFR_MS_PARAMS["lgec_yhi"],
    k_trans_c0=MEAN_SFR_MS_PARAMS["k_trans_c0"],
    a_late_x0=MEAN_SFR_MS_PARAMS["a_late_x0"],
    a_late_k=MEAN_SFR_MS_PARAMS["a_late_k"],
    a_late_ylo=MEAN_SFR_MS_PARAMS["a_late_ylo"],
    a_late_yhi=MEAN_SFR_MS_PARAMS["a_late_yhi"],
    fms_logtc_x0=MEAN_Q_PARAMS["fms_logtc_x0"],
    fms_logtc_k=MEAN_Q_PARAMS["fms_logtc_k"],
    fms_logtc_ylo=MEAN_Q_PARAMS["fms_logtc_ylo"],
    fms_logtc_yhi=MEAN_Q_PARAMS["fms_logtc_yhi"],
    fms_late_x0=MEAN_Q_PARAMS["fms_late_x0"],
    fms_late_k=MEAN_Q_PARAMS["fms_late_k"],
    fms_late_ylo=MEAN_Q_PARAMS["fms_late_ylo"],
    fms_late_yhi=MEAN_Q_PARAMS["fms_late_yhi"],
):
    """Star formation rate and stellar mass as a function of time
    averaged over centrals living in halos with present-day mass logm0.

    Parameters
    ----------
    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

        The size n should be large enough so that the log_sm integration
        can be accurately calculated with the midpoint rule.
        Typically n >~100 is sufficient for most purposes.

    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    t0 : float, optional
        Age of the universe in Gyr at the time halo mass attains the input logm0.
        There must exist some entry of the input cosmic_time array within 50Myr of t0.
        Default is ~13.85 Gyr.

    **mean_mah_params : floats, optional
        Any keyword of halo_assembly.MEAN_MAH_PARAMS is accepted

    **mean_sfr_ms_params : floats, optional
        Any keyword of main_sequence_sfr_eff.MEAN_SFR_MS_PARAMS is accepted

    **mean_q_params : floats, optional
        Any keyword of quenching_history.MEAN_Q_PARAMS is accepted

    Returns
    -------
    log_sfr : ndarray of shape (n, )
        Base-10 log of <SFR|M0,t> in units of Msun/yr

    log_sm : ndarray of shape (n, )
        Base-10 log of <M*|M0,t> in units of Msun

    Notes
    -----
    If you only need to predict SFR or M* at a small handful of redshifts,
    you should use this function to build an interpolation table with n>~100

    """
    logm0, logt, dtarr, indx_t0 = _process_halo_mah_args(logm0, cosmic_time, t0)

    mean_mah_params = jax_np.array(
        (
            dmhdt_x0_c0,
            dmhdt_x0_c1,
            dmhdt_k_c0,
            dmhdt_k_c1,
            dmhdt_ylo_c0,
            dmhdt_ylo_c1,
            dmhdt_yhi_c0,
            dmhdt_yhi_c1,
        )
    ).astype("f4")

    mean_sfr_ms_params = jax_np.array(
        (
            lge0_lgmc,
            lge0_at_lgmc,
            lge0_early_slope,
            lge0_late_slope,
            k_early_x0,
            k_early_k,
            k_early_ylo,
            k_early_yhi,
            lgtc_x0,
            lgtc_k,
            lgtc_ylo,
            lgtc_yhi,
            lgec_x0,
            lgec_k,
            lgec_ylo,
            lgec_yhi,
            k_trans_c0,
            a_late_x0,
            a_late_k,
            a_late_ylo,
            a_late_yhi,
        )
    ).astype("f4")

    mean_q_params = jax_np.array(
        (
            fms_logtc_x0,
            fms_logtc_k,
            fms_logtc_ylo,
            fms_logtc_yhi,
            fms_late_x0,
            fms_late_k,
            fms_late_ylo,
            fms_late_yhi,
        )
    ).astype("f4")

    log_sfr, log_smh = _mean_log_mstar_history_jax_kern(
        logt, dtarr, logm0, mean_mah_params, mean_sfr_ms_params, mean_q_params, indx_t0
    )

    return np.array(log_sfr), np.array(log_smh)


def _mean_log_mstar_history_jax_kern(
    logt, dtarr, logm0, mean_mah_params, mean_sfr_eff_params, mean_q_params, indx_t0
):
    log_sfr = _mean_log_sfr_history_jax_kern(
        logt, dtarr, logm0, mean_mah_params, mean_sfr_eff_params, mean_q_params, indx_t0
    )
    log_smh = _calculate_cumulative_in_situ_mass(log_sfr, dtarr)
    return log_sfr, log_smh


def _mean_log_sfr_history_jax_kern(
    logt, dtarr, logm0, mean_mah_params, mean_sfr_eff_params, mean_q_params, indx_t0
):
    _x = _mean_halo_assembly_jax_kern(logt, dtarr, logm0, *mean_mah_params, indx_t0)
    log_dmbdt = jax_np.log10(FB) + _x[1]
    log_sfr_eff_ms = mean_log_sfr_efficiency_ms_jax(logt, logm0, *mean_sfr_eff_params)
    log_frac_ms = _mean_log_main_sequence_fraction(logt, logm0, *mean_q_params)
    log_sfr = log_dmbdt + log_sfr_eff_ms + log_frac_ms
    return log_sfr


def _calculate_cumulative_in_situ_mass(log_sfr, dtarr):
    log_smh = jax_np.log10(jax_np.cumsum(jax_np.power(10, log_sfr)) * dtarr) + 9.0
    return log_smh
