"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from . import mc_diffmahpop_kernels as mcdk

T_OBS_FIT_MIN = 0.5
T_TARGET_VAR_MIN = 2.0
LGSMAH_MIN = -15.0


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _wmse(x, y, w):
    d = y - x
    return jnp.average(d * d, weights=w)


@jjit
def predict_mah_targets_singlebin(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    _res = mcdk._mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    (
        mah_params_tpt0,
        mah_params_tp,
        t_peak,
        ftpt0,
        mc_tpt0,
        dmhdt_tpt0,
        log_mah_tpt0,
        dmhdt_tp,
        log_mah_tp,
    ) = _res

    f = ftpt0.reshape((-1, 1))
    delta_log_mah_tpt0 = log_mah_tpt0 - lgm_obs
    delta_log_mah_tp = log_mah_tp - lgm_obs
    mean_delta_log_mah = jnp.mean(
        f * delta_log_mah_tpt0 + (1 - f) * delta_log_mah_tp, axis=0
    )

    std_log_mah = jnp.std(f * log_mah_tpt0 + (1 - f) * log_mah_tp, axis=0)

    dmhdt_tpt0 = jnp.clip(dmhdt_tpt0, 10**LGSMAH_MIN)  # make log-safe
    dmhdt_tp = jnp.clip(dmhdt_tp, 10**LGSMAH_MIN)  # make log-safe

    lgsmah_tpt0 = jnp.log10(dmhdt_tpt0) - log_mah_tpt0  # compute lgsmah
    lgsmah_tpt0 = jnp.clip(lgsmah_tpt0, LGSMAH_MIN)  # impose lgsMAH clip

    lgsmah_tp = jnp.log10(dmhdt_tp) - log_mah_tpt0  # compute lgsmah
    lgsmah_tp = jnp.clip(lgsmah_tp, LGSMAH_MIN)  # impose lgsMAH clip

    weights_ftpt0 = jnp.concatenate((ftpt0, 1 - ftpt0))
    lgsmah_pred = jnp.concatenate((lgsmah_tpt0, lgsmah_tp))

    frac_peaked = jnp.average(lgsmah_pred == LGSMAH_MIN, axis=0, weights=weights_ftpt0)

    return mean_delta_log_mah, std_log_mah, frac_peaked


@jjit
def _loss_mah_moments_singlebin(
    diffmahpop_params,
    tarr,
    lgm_obs,
    t_obs,
    ran_key,
    lgt0,
    target_mean_delta_log_mah,
    target_std_log_mah,
    target_frac_peaked,
):
    _preds = predict_mah_targets_singlebin(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    mean_delta_log_mah, std_log_mah, frac_peaked = _preds
    loss = _mse(mean_delta_log_mah, target_mean_delta_log_mah)

    msk_std = tarr < T_TARGET_VAR_MIN
    std_weights = jnp.where(msk_std, 0.0, 1.0)
    loss = loss + _wmse(std_log_mah, target_std_log_mah, std_weights)
    loss = loss + _mse(frac_peaked, target_frac_peaked)
    return loss
