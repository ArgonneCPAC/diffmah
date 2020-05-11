"""
"""
from jax import numpy as jax_np
from .quenching_history import _mean_log_main_sequence_fraction
from .sfr_efficiency import mean_log_sfr_efficiency_ms_jax
from .halo_assembly import _mean_halo_assembly_function

FB = 0.158


def _mean_log_sfr_history(
    mean_mah_params, mean_sfr_eff_params, q_params, logm0, logt, indx_t0
):
    tarr = jax_np.power(10, logt)
    logt0 = logt[indx_t0]

    logmah, log_dmhdt = _mean_halo_assembly_function(
        mean_mah_params, tarr, logm0, indx_t0, logt0
    )
    log_dmbdt = jax_np.log10(FB) + log_dmhdt + 9.0

    log_sfr_eff_ms = mean_log_sfr_efficiency_ms_jax(logm0, logt, mean_sfr_eff_params)

    q_data = logm0, logt
    log_sfr_eff_ms = _mean_log_main_sequence_fraction(q_params, q_data)

    log_sfr = log_dmbdt + log_sfr_eff_ms + log_sfr_eff_ms
    return log_sfr
