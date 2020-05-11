"""
"""
from jax import numpy as jax_np
from .quenching_history import _mean_log_main_sequence_fraction
from .sfr_efficiency import mean_log_sfr_efficiency_ms_jax
from .halo_assembly import _mean_halo_assembly_function

FB = 0.158


def _mean_log_mstar_history_jax_kern(
    mean_mah_params,
    mean_sfr_eff_params,
    q_params,
    logm0,
    logt_table,
    indx_t0,
    dt,
    indx_pred,
):
    log_sfr_table = _mean_log_sfr_history_jax_kern(
        mean_mah_params, mean_sfr_eff_params, q_params, logm0, logt_table, indx_t0
    )
    log_smh_table = (
        jax_np.log10(jax_np.cumsum(jax_np.power(10, log_sfr_table)) * dt) + 9
    )
    return log_sfr_table[indx_pred], log_smh_table[indx_pred]


def _mean_log_sfr_history_jax_kern(
    mean_mah_params, mean_sfr_eff_params, q_params, logm0, logt, indx_t0
):
    tarr = jax_np.power(10, logt)
    logt0 = logt[indx_t0]

    logmah, log_dmhdt = _mean_halo_assembly_function(
        mean_mah_params, tarr, logm0, indx_t0, logt0
    )
    log_dmbdt = jax_np.log10(FB) + log_dmhdt + 9.0
    log_sfr_eff_ms = mean_log_sfr_efficiency_ms_jax(mean_sfr_eff_params, logm0, logt)
    log_frac_ms = _mean_log_main_sequence_fraction(q_params, logm0, logt)
    log_sfr = log_dmbdt + log_sfr_eff_ms + log_frac_ms
    return log_sfr
