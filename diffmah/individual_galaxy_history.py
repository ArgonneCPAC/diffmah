"""Model for individual galaxy mass assembly."""
from jax import numpy as jnp
from jax import jit as jjit
from .individual_halo_history import _calc_halo_history
from .epsilon_sfr_emerge import _sfr_efficiency_kernel
from .galaxy_quenching import _quenching_kern


FB = 0.156


@jjit
def _sm_history(lgt, z, logtmp, logmp, x0, k, lge, u_dy, sfr_ms_params, q_params, dt):
    _a = lgt, z, logtmp, logmp, x0, k, lge, u_dy, sfr_ms_params, q_params
    dmhdt, log_mah, main_sequence_sfr, sfr = _sfr_history(*_a)
    mstar = _integrate_sfr(sfr, dt)
    return dmhdt, log_mah, main_sequence_sfr, sfr, mstar


@jjit
def _sfr_history(lgt, z, logtmp, logmp, x0, k, lge, u_dy, sfr_ms_params, q_params):
    _a = lgt, z, logtmp, logmp, x0, k, lge, u_dy, sfr_ms_params
    dmhdt, log_mah, main_sequence_sfr = _main_sequence_history(*_a)
    qfrac = _quenching_kern(lgt, *q_params)
    sfr = qfrac * main_sequence_sfr
    return dmhdt, log_mah, main_sequence_sfr, sfr


@jjit
def _main_sequence_history(lgt, z, logtmp, logmp, x0, k, lge, u_dy, sfr_ms_params):
    dmhdt, log_mah = _calc_halo_history(lgt, logtmp, logmp, x0, k, lge, u_dy)
    sfr_eff_data = (10 ** log_mah, z)
    efficiency = _sfr_efficiency_kernel(sfr_ms_params, sfr_eff_data)
    main_sequence_sfr = FB * efficiency * dmhdt
    return dmhdt, log_mah, main_sequence_sfr


@jjit
def _integrate_sfr(sfr, dt):
    return jnp.cumsum(sfr * dt) * 1e9
