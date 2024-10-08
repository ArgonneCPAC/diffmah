"""
"""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from . import mc_diffmahpop_kernels_monocens_fixed_tpeak as mcdk
from .diffmahpop_params_monocensat import (
    DEFAULT_DIFFMAHPOP_U_PARAMS,
    get_diffmahpop_params_from_u_params,
)

T_OBS_FIT_MIN = 0.5


def get_varied_u_params():
    fixed_u_pnames = (
        "u_cen_tp_x0_ylo",
        "u_cen_tp_x0_yhi",
        "u_utp_loc_lgm_ylo_t0",
        "u_utp_loc_lgm_ylo_early",
        "u_utp_loc_lgm_ylo_late",
        "u_utp_loc_lgm_x0",
        "u_utp_scale_lgm_ylo_t0",
        "u_utp_scale_lgm_ylo_early",
        "u_utp_scale_lgm_ylo_late",
    )
    u_pdict = dict()
    gen = zip(DEFAULT_DIFFMAHPOP_U_PARAMS._fields, DEFAULT_DIFFMAHPOP_U_PARAMS)
    for key, val in gen:
        if key not in fixed_u_pnames:
            u_pdict[key] = val
    VariedUParams = namedtuple("VariedUParams", u_pdict.keys())
    varied_u_params = VariedUParams(**u_pdict)
    return varied_u_params


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_mah_moments_singlebin_u_params(
    varied_u_params,
    tarr,
    lgm_obs,
    t_obs,
    t_peak_sample,
    ran_key,
    lgt0,
    target_mean_log_mah,
    target_std_log_mah,
    target_frac_peaked,
):
    u_params = DEFAULT_DIFFMAHPOP_U_PARAMS._replace(**varied_u_params._asdict())
    diffmahpop_params = get_diffmahpop_params_from_u_params(u_params)
    _preds = mcdk.predict_mah_moments_singlebin(
        diffmahpop_params, tarr, lgm_obs, t_obs, t_peak_sample, ran_key, lgt0
    )
    mean_log_mah, std_log_mah, frac_peaked = _preds
    loss = _mse(mean_log_mah, target_mean_log_mah)
    loss = loss + _mse(std_log_mah, target_std_log_mah)
    # loss = loss + _mse(frac_peaked, target_frac_peaked)
    return loss


_U = (None, 0, 0, 0, 0, 0, None, 0, 0, 0)
_loss_mah_moments_multibin_vmap = jjit(
    vmap(_loss_mah_moments_singlebin_u_params, in_axes=_U)
)


@jjit
def _loss_mah_moments_multibin_kern(
    diffmahpop_params,
    tarr_matrix,
    lgm_obs_arr,
    t_obs_arr,
    t_peak_arr,
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
        t_peak_arr,
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
    t_peak_arr,
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
        t_peak_arr,
        ran_key,
        lgt0,
        target_mean_log_mahs,
        target_std_log_mahs,
        target_frac_peaked,
    )
    return jnp.mean(losses)


_loss_and_grads_mah_moments_multibin = value_and_grad(loss_mah_moments_multibin)


@jjit
def loss_and_grads_mah_moments_multibin(params, loss_data):
    return _loss_and_grads_mah_moments_multibin(params, *loss_data)
