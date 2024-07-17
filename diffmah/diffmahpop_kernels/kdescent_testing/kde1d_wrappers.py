"""
"""

from jax import random as jran
from jax import value_and_grad, vmap

try:
    import kdescent
except ImportError:
    pass
from jax import jit as jjit
from jax import numpy as jnp

from .. import diffmahpop_params as dpp
from .. import mc_diffmahpop_kernels as mdk

N_T_PER_BIN = 5


@jjit
def mc_diffmah_preds(diffmahpop_u_params, pred_data):
    diffmahpop_params = dpp.get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    tarr, lgm_obs, t_obs, ran_key, lgt0 = pred_data
    _res = mdk._mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    ftpt0 = _res[3]
    log_mah_tpt0 = _res[6]
    log_mah_tp = _res[8]
    return log_mah_tpt0, log_mah_tp, ftpt0


@jjit
def single_sample_kde_loss_self_fit(
    diffmahpop_u_params,
    tarr,
    lgm_obs,
    t_obs,
    ran_key,
    lgt0,
    log_mahs_target,
    weights_target,
):
    s = (-1, 1)
    kcalc0 = kdescent.KCalc(log_mahs_target[:, 0].reshape(s), weights_target)
    kcalc1 = kdescent.KCalc(log_mahs_target[:, 1].reshape(s), weights_target)
    kcalc2 = kdescent.KCalc(log_mahs_target[:, 2].reshape(s), weights_target)
    kcalc3 = kdescent.KCalc(log_mahs_target[:, 3].reshape(s), weights_target)
    kcalc4 = kdescent.KCalc(log_mahs_target[:, 4].reshape(s), weights_target)

    ran_key, pred_key = jran.split(ran_key, 2)
    pred_data = tarr, lgm_obs, t_obs, pred_key, lgt0
    _res = mc_diffmah_preds(diffmahpop_u_params, pred_data)
    log_mah_tpt0, log_mah_tp, ftpt0 = _res

    log_mahs_pred = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    weights_pred = jnp.concatenate((ftpt0, 1 - ftpt0))

    kcalc_keys = jran.split(ran_key, N_T_PER_BIN)

    model_counts0, truth_counts0 = kcalc0.compare_kde_counts(
        kcalc_keys[0], log_mahs_pred[:, 0].reshape(s), weights_pred
    )
    model_counts1, truth_counts1 = kcalc1.compare_kde_counts(
        kcalc_keys[1], log_mahs_pred[:, 1].reshape(s), weights_pred
    )
    model_counts2, truth_counts2 = kcalc2.compare_kde_counts(
        kcalc_keys[2], log_mahs_pred[:, 2].reshape(s), weights_pred
    )
    model_counts3, truth_counts3 = kcalc3.compare_kde_counts(
        kcalc_keys[3], log_mahs_pred[:, 3].reshape(s), weights_pred
    )
    model_counts4, truth_counts4 = kcalc4.compare_kde_counts(
        kcalc_keys[4], log_mahs_pred[:, 4].reshape(s), weights_pred
    )

    diff0 = model_counts0 - truth_counts0
    diff1 = model_counts1 - truth_counts1
    diff2 = model_counts2 - truth_counts2
    diff3 = model_counts3 - truth_counts3
    diff4 = model_counts4 - truth_counts4

    loss0 = jnp.mean(diff0**2)
    loss1 = jnp.mean(diff1**2)
    loss2 = jnp.mean(diff2**2)
    loss3 = jnp.mean(diff3**2)
    loss4 = jnp.mean(diff4**2)

    loss = loss0 + loss1 + loss2 + loss3 + loss4
    return loss


single_sample_kde_loss_and_grad_self_fit = jjit(
    value_and_grad(single_sample_kde_loss_self_fit)
)


@jjit
def get_single_sample_self_fit_target_data(
    u_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    pred_data = tarr, lgm_obs, t_obs, ran_key, lgt0
    _res = mc_diffmah_preds(u_params, pred_data)
    log_mah_tpt0, log_mah_tp, ftpt0 = _res
    log_mahs_target = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    weights_target = jnp.concatenate((ftpt0, 1 - ftpt0))
    return log_mahs_target, weights_target


_A = (None, 0, 0, 0, 0, None)
get_multisample_self_fit_target_data = jjit(
    vmap(get_single_sample_self_fit_target_data, in_axes=_A)
)

_L = (None, 0, 0, 0, 0, None, 0, 0)
_multisample_kde_loss_self_fit = jjit(vmap(single_sample_kde_loss_self_fit, in_axes=_L))


@jjit
def multisample_kde_loss_self_fit(
    diffmahpop_u_params,
    tarr_matrix,
    lgmobsarr,
    tobsarr,
    ran_keys,
    lgt0,
    log_mahs_targets,
    weights_targets,
):
    losses = _multisample_kde_loss_self_fit(
        diffmahpop_u_params,
        tarr_matrix,
        lgmobsarr,
        tobsarr,
        ran_keys,
        lgt0,
        log_mahs_targets,
        weights_targets,
    )
    loss = jnp.mean(losses)
    return loss


multisample_kde_loss_and_grad_self_fit = jjit(
    value_and_grad(multisample_kde_loss_self_fit)
)
