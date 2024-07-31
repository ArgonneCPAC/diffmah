"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from . import mc_diffmahpop_kernels as mcdk

T_OBS_FIT_MIN = 0.5


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_mah_moments_singlebin_cens(
    diffmahpop_params,
    tarr,
    lgm_obs,
    t_obs,
    ran_key,
    lgt0,
    target_mean_log_mah,
    target_std_log_mah,
):
    _preds = mcdk.predict_mah_moments_singlebin(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    mean_log_mah, std_log_mah = _preds
    loss = _mse(mean_log_mah, target_mean_log_mah)
    loss = loss + _mse(std_log_mah, target_std_log_mah)
    return loss
