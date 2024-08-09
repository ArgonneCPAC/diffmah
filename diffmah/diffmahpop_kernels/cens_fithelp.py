"""
"""

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from ..diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop
from . import diffmahpop_params as dpp
from . import mc_diffmahpop_kernels as mcdk

T_OBS_FIT_MIN = 1.5
T_TARGET_VAR_MIN = 2.0
T_TARGET_VAR_MAX = 0.1
LGSMAH_MIN = -15.0
EPS = 1e-3


@jjit
def _get_var_weights(tarr, t_obs):
    msk_std = tarr < T_TARGET_VAR_MIN
    msk_std |= tarr > t_obs - T_TARGET_VAR_MAX
    var_weights = jnp.where(msk_std, 0.0, 1.0)
    return var_weights


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

    std_weights = _get_var_weights(tarr, t_obs)
    loss = loss + _wmse(std_log_mah, target_std_log_mah, std_weights)
    # loss = loss + _mse(frac_peaked, target_frac_peaked)
    return loss


_A = (None, 0, 0, 0, 0, None, 0, 0, 0)
_loss_mah_moments_multibin = jjit(vmap(_loss_mah_moments_singlebin, in_axes=_A))


@jjit
def loss_mah_moments_multibin(
    diffmahpop_params,
    tarr_matrix,
    lgm_obs_arr,
    t_obs_arr,
    ran_keys,
    lgt0,
    target_mean_delta_log_mah_matrix,
    target_std_log_mah_matrix,
    target_frac_peaked_matrix,
):
    losses = _loss_mah_moments_multibin(
        diffmahpop_params,
        tarr_matrix,
        lgm_obs_arr,
        t_obs_arr,
        ran_keys,
        lgt0,
        target_mean_delta_log_mah_matrix,
        target_std_log_mah_matrix,
        target_frac_peaked_matrix,
    )
    return jnp.mean(losses)


@jjit
def global_loss_kern(diffmahpop_u_params, loss_data):
    diffmahpop_params = dpp.get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    return loss_mah_moments_multibin(diffmahpop_params, *loss_data)


global_loss_and_grad_kern = jjit(value_and_grad(global_loss_kern))


def compute_targets_singlebin(halo_samples, t_obs_samples, lgm_obs, t_obs, lgt0):
    it_obs = np.argmin(np.abs(t_obs_samples - t_obs))
    t_obs = t_obs_samples[it_obs]

    cens = halo_samples[it_obs]
    lgm_obs_arr = np.sort(np.unique(cens["lgm_obs"]))
    mah_keys = ("logm0", "logtc", "early_index", "late_index")
    mah_params = DEFAULT_MAH_PARAMS._make([cens[key] for key in mah_keys])
    ilgm_obs = np.argmin(np.abs(lgm_obs_arr - lgm_obs))
    lgm_obs = lgm_obs_arr[ilgm_obs]
    mmsk = cens["lgm_obs"] == lgm_obs
    mah_params_target = DEFAULT_MAH_PARAMS._make([x[mmsk] for x in mah_params])
    t_peak_target = cens["t_peak"][mmsk]
    tarr = np.linspace(T_OBS_FIT_MIN, t_obs - EPS, 50)
    dmhdt, log_mah = mah_halopop(mah_params_target, tarr, t_peak_target, lgt0)

    lgm_obs_sample = mah_halopop(
        mah_params_target, np.zeros(1) + t_obs, t_peak_target, lgt0
    )[1][:, 0]
    log_mah_rescaled = log_mah - (lgm_obs_sample.reshape((-1, 1)) - lgm_obs)

    delta_log_mah = log_mah_rescaled - lgm_obs
    mean_delta_log_mah = np.mean(delta_log_mah, axis=0)
    std_log_mah = np.std(log_mah_rescaled, axis=0)

    dmhdt = jnp.clip(dmhdt, 10**LGSMAH_MIN)  # make log-safe
    lgsmah = jnp.log10(dmhdt) - log_mah
    lgsmah = jnp.clip(lgsmah, LGSMAH_MIN)

    frac_peaked = np.mean(lgsmah == LGSMAH_MIN, axis=0)

    return lgm_obs, t_obs, tarr, mean_delta_log_mah, std_log_mah, frac_peaked


def get_random_target_collection(
    halo_samples, t_obs_samples, lgt0, ran_key, n_targets=100
):
    ran_key, lgm_obs_key, t_obs_key = jran.split(ran_key, 3)

    target_collector = []
    for __ in range(n_targets):
        lgm_obs = jran.uniform(lgm_obs_key, minval=11, maxval=15, shape=())
        t_obs = jran.uniform(t_obs_key, minval=4, maxval=13.5, shape=())
        targets = compute_targets_singlebin(
            halo_samples, t_obs_samples, lgm_obs, t_obs, lgt0
        )
        target_collector.append(targets)
    lgm_obs_arr = np.array([x[0] for x in target_collector])
    t_obs_arr = np.array([x[1] for x in target_collector])
    tarr_matrix = np.array([x[2] for x in target_collector])
    mean_delta_log_mah_matrix = np.array([x[3] for x in target_collector])
    std_log_mahs_matrix = np.array([x[4] for x in target_collector])
    frac_peaked_matrix = np.array([x[5] for x in target_collector])

    return (
        lgm_obs_arr,
        t_obs_arr,
        tarr_matrix,
        mean_delta_log_mah_matrix,
        std_log_mahs_matrix,
        frac_peaked_matrix,
    )
