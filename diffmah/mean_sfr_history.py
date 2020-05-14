"""Module implementing the get_mean_galaxy_history function."""
import numpy as np
from jax import numpy as jax_np
from .halo_assembly import _mean_halo_assembly_jax_kern, TODAY
from .halo_assembly import MEAN_MAH_PARAMS, _get_dt_array
from .main_sequence_sfr_eff import mean_log_sfr_efficiency_ms_jax, MEAN_SFR_MS_PARAMS
from .quenching_history import _mean_log_main_sequence_fraction, MEAN_Q_PARAMS
from .utils import _get_param_dict

FB = 0.158
T_TABLE = np.linspace(0.1, TODAY, 250)


def get_mean_galaxy_history(logm0, cosmic_time, t_table=T_TABLE, **kwargs):
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
    logt_table, indx_t0, dtarr, indx_pred = _process_args(cosmic_time, t_table)

    _x = _get_all_params(logm0, **kwargs)
    mean_mah_params, mean_sfr_ms_params, mean_q_params = _x

    galaxy_history = _mean_log_mstar_history_jax_kern(
        mean_mah_params,
        mean_sfr_ms_params,
        mean_q_params,
        logm0,
        logt_table,
        indx_t0,
        dtarr,
        indx_pred,
    )

    return galaxy_history


def _process_args(cosmic_time, t_table):
    logt_table = np.log10(t_table)
    indx_t0 = np.argmin(np.abs(t_table - TODAY))
    dtarr = _get_dt_array(t_table)

    _indx_pred = [np.argmin(np.abs(t_table - t)) for t in cosmic_time]
    indx_pred = np.array(_indx_pred).astype("i4")
    return logt_table, indx_t0, dtarr, indx_pred


def _get_all_params(logm0, **kwargs):
    mean_mah_params = _get_param_ndarray(MEAN_MAH_PARAMS, **kwargs)
    mean_sfr_ms_params = _get_param_ndarray(MEAN_SFR_MS_PARAMS, **kwargs)
    mean_q_params = _get_param_ndarray(MEAN_Q_PARAMS, **kwargs)
    return mean_mah_params, mean_sfr_ms_params, mean_q_params


def _get_param_ndarray(defaults, **kwargs):
    param_dict = _get_param_dict(defaults, strict=False, **kwargs)
    return jax_np.array(list(param_dict.values()))


def _mean_log_mstar_history_jax_kern(
    mean_mah_params,
    mean_sfr_eff_params,
    mean_q_params,
    logm0,
    logt_table,
    indx_t0,
    dtarr,
    indx_pred,
):
    log_sfr_table = _mean_log_sfr_history_jax_kern(
        mean_mah_params,
        mean_sfr_eff_params,
        mean_q_params,
        logm0,
        logt_table,
        dtarr,
        indx_t0,
    )
    log_smh_table = jax_np.log10(jax_np.cumsum(jax_np.power(10, log_sfr_table)) * dtarr)
    return log_sfr_table[indx_pred] - 9, log_smh_table[indx_pred]


def _mean_log_sfr_history_jax_kern(
    mean_mah_params, mean_sfr_eff_params, mean_q_params, logm0, logt, dtarr, indx_t0
):
    logmah, log_dmhdt = _mean_halo_assembly_jax_kern(
        mean_mah_params, logm0, logt, dtarr, indx_t0
    )

    log_dmbdt = jax_np.log10(FB) + log_dmhdt + 9.0
    log_sfr_eff_ms = mean_log_sfr_efficiency_ms_jax(mean_sfr_eff_params, logm0, logt)
    log_frac_ms = _mean_log_main_sequence_fraction(mean_q_params, logm0, logt)
    log_sfr = log_dmbdt + log_sfr_eff_ms + log_frac_ms
    return log_sfr
