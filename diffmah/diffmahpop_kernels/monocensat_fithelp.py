"""
"""

from jax import jit as jjit
from jax import value_and_grad

from . import monocens_fithelp, monosats_fithelp


@jjit
def loss_mah_moments_multibin_censat(
    diffmahpop_params,
    tarr_matrix,
    lgm_obs_arr,
    t_obs_arr,
    ran_key,
    lgt0,
    target_mean_log_mahs_cens,
    target_std_log_mahs_cens,
    target_frac_peaked_cens,
    target_mean_log_mahs_sats,
    target_std_log_mahs_sats,
    target_frac_peaked_sats,
):
    loss_cens = monocens_fithelp.loss_mah_moments_multibin(
        diffmahpop_params,
        tarr_matrix,
        lgm_obs_arr,
        t_obs_arr,
        ran_key,
        lgt0,
        target_mean_log_mahs_cens,
        target_std_log_mahs_cens,
        target_frac_peaked_cens,
    )

    loss_sats = monosats_fithelp.loss_mah_moments_multibin(
        diffmahpop_params,
        tarr_matrix,
        lgm_obs_arr,
        t_obs_arr,
        ran_key,
        lgt0,
        target_mean_log_mahs_sats,
        target_std_log_mahs_sats,
        target_frac_peaked_sats,
    )
    return loss_cens + loss_sats


loss_and_grads_mah_moments_multibin_censat = jjit(
    value_and_grad(loss_mah_moments_multibin_censat)
)
