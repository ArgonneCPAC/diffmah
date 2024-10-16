"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from . import mc_bimod_sats as mcs

T_OBS_FIT_MIN = 0.5


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_mah_moments_singlebin(
    diffmahpop_params,
    tarr,
    lgm_obs,
    t_obs,
    ran_key,
    lgt0,
    target_mean_log_mah,
    target_std_log_mah,
    target_frac_peaked,
):
    _preds = mcs.predict_mah_moments_singlebin(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    mean_log_mah, std_log_mah, frac_peaked = _preds
    loss = _mse(mean_log_mah, target_mean_log_mah)
    loss = loss + _mse(std_log_mah, target_std_log_mah)
    # loss = loss + _mse(frac_peaked, target_frac_peaked)
    return loss


_U = (None, 0, 0, 0, 0, None, 0, 0, 0)
_loss_mah_moments_multibin_vmap = jjit(vmap(_loss_mah_moments_singlebin, in_axes=_U))


@jjit
def _loss_mah_moments_multibin_kern(
    diffmahpop_params,
    tarr_matrix,
    lgm_obs_arr,
    t_obs_arr,
    ran_key,
    lgt0,
    target_mean_log_mahs,
    target_std_log_mahs,
    target_frac_peaked,
):
    ran_keys = jran.split(ran_key, tarr_matrix.shape[0])
    return _loss_mah_moments_multibin_vmap(
        diffmahpop_params,
        tarr_matrix,
        lgm_obs_arr,
        t_obs_arr,
        ran_keys,
        lgt0,
        target_mean_log_mahs,
        target_std_log_mahs,
        target_frac_peaked,
    )


@jjit
def loss_mah_moments_multibin(
    diffmahpop_params,
    tarr_matrix,
    lgm_obs_arr,
    t_obs_arr,
    ran_key,
    lgt0,
    target_mean_log_mahs,
    target_std_log_mahs,
    target_frac_peaked,
):
    losses = _loss_mah_moments_multibin_kern(
        diffmahpop_params,
        tarr_matrix,
        lgm_obs_arr,
        t_obs_arr,
        ran_key,
        lgt0,
        target_mean_log_mahs,
        target_std_log_mahs,
        target_frac_peaked,
    )
    return jnp.mean(losses)


loss_and_grads_mah_moments_multibin = value_and_grad(loss_mah_moments_multibin)
