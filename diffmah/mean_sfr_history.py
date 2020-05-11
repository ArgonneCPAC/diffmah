"""
"""
from jax import numpy as jax_np
from .halo_assembly import _mean_halo_assembly_function
from .halo_assembly import MEAN_MAH_PARAMS, _get_individual_mah_params
from .sfr_efficiency import mean_log_sfr_efficiency_ms_jax, MEDIAN_SFR_MS_PARAMS
from .quenching_history import _mean_log_main_sequence_fraction, MEAN_Q_PARAMS
from .utils import _get_param_dict

FB = 0.158


def get_mean_galaxy_history(logm0, cosmic_time, **kwargs):
    """Star formation rate and stellar mass as a function of time
    averaged over centrals living in halos with present-day mass logm0.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    Returns
    -------
    log_sfr : ndarray of shape (n, )
        Base-10 log of SFR in units of Msun/yr

    log_sm : ndarray of shape (n, )
        Base-10 log of in-situ stellar mass in units of Msun

    """
    raise NotImplementedError()


def _get_all_param_dicts(logm0, **kwargs):
    mean_mah_param_dict = _get_param_dict(MEAN_MAH_PARAMS, strict=False)
    mean_sfr_param_dict = _get_param_dict(MEDIAN_SFR_MS_PARAMS, strict=False, **kwargs)
    mean_q_param_dict = _get_param_dict(MEAN_Q_PARAMS, strict=False, **kwargs)

    _x = jax_np.zeros(list(mean_mah_param_dict.values()))
    mah_param_dict = _get_individual_mah_params(_x, logm0)
    return mah_param_dict, mean_sfr_param_dict, mean_q_param_dict


def _mean_log_mstar_history_jax_kern(
    mean_mah_params,
    mean_sfr_eff_params,
    mean_q_params,
    logm0,
    logt_table,
    indx_t0,
    dt,
    indx_pred,
):
    log_sfr_table = _mean_log_sfr_history_jax_kern(
        mean_mah_params, mean_sfr_eff_params, mean_q_params, logm0, logt_table, indx_t0
    )
    log_smh_table = (
        jax_np.log10(jax_np.cumsum(jax_np.power(10, log_sfr_table)) * dt) + 9
    )
    return log_sfr_table[indx_pred], log_smh_table[indx_pred]


def _mean_log_sfr_history_jax_kern(
    mean_mah_params, mean_sfr_eff_params, mean_q_params, logm0, logt, indx_t0
):
    tarr = jax_np.power(10, logt)
    logt0 = logt[indx_t0]

    logmah, log_dmhdt = _mean_halo_assembly_function(
        mean_mah_params, tarr, logm0, indx_t0, logt0
    )
    log_dmbdt = jax_np.log10(FB) + log_dmhdt + 9.0
    log_sfr_eff_ms = mean_log_sfr_efficiency_ms_jax(mean_sfr_eff_params, logm0, logt)
    log_frac_ms = _mean_log_main_sequence_fraction(mean_q_params, logm0, logt)
    log_sfr = log_dmbdt + log_sfr_eff_ms + log_frac_ms
    return log_sfr
