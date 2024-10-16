"""
"""

from jax import jit as jjit
from jax import random as jran
from jax import value_and_grad

from . import bimod_cens_fithelp, bimod_sats_fithelp
from .bimod_censat_params import (
    DEFAULT_DIFFMAHPOP_PARAMS,
    DEFAULT_DIFFMAHPOP_U_PARAMS,
    get_diffmahpop_params_from_u_params,
)


@jjit
def loss_mah_moments_multibin_censat(
    varied_diffmahpop_params,
    tarr_matrix_cens,
    lgm_obs_arr_cens,
    t_obs_arr_cens,
    tarr_matrix_sats,
    lgm_obs_arr_sats,
    t_obs_arr_sats,
    ran_key,
    lgt0,
    target_mean_log_mahs_cens,
    target_std_log_mahs_cens,
    target_frac_peaked_cens,
    target_mean_log_mahs_sats,
    target_std_log_mahs_sats,
    target_frac_peaked_sats,
):
    diffmahpop_params = DEFAULT_DIFFMAHPOP_PARAMS._replace(
        **varied_diffmahpop_params._asdict()
    )
    ran_key_cens, ran_key_sats = jran.split(ran_key, 2)
    loss_cens = bimod_cens_fithelp.loss_mah_moments_multibin(
        diffmahpop_params,
        tarr_matrix_cens,
        lgm_obs_arr_cens,
        t_obs_arr_cens,
        ran_key_cens,
        lgt0,
        target_mean_log_mahs_cens,
        target_std_log_mahs_cens,
        target_frac_peaked_cens,
    )

    loss_sats = bimod_sats_fithelp.loss_mah_moments_multibin(
        diffmahpop_params,
        tarr_matrix_sats,
        lgm_obs_arr_sats,
        t_obs_arr_sats,
        ran_key_sats,
        lgt0,
        target_mean_log_mahs_sats,
        target_std_log_mahs_sats,
        target_frac_peaked_sats,
    )
    return loss_cens + loss_sats


loss_and_grads_mah_moments_multibin_censat = jjit(
    value_and_grad(loss_mah_moments_multibin_censat)
)


@jjit
def loss_mah_moments_multibin_censat_u_params(u_params, loss_data):
    u_params = DEFAULT_DIFFMAHPOP_U_PARAMS._replace(**u_params._asdict())
    params = get_diffmahpop_params_from_u_params(u_params)
    return loss_mah_moments_multibin_censat(params, *loss_data)


loss_and_grads_mah_moments_multibin_censat_u_params = jjit(
    value_and_grad(loss_mah_moments_multibin_censat_u_params)
)
