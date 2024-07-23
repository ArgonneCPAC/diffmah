"""
"""

from jax import random as jran
from jax import value_and_grad, vmap

from ...diffmah_kernels import mah_halopop

try:
    import kdescent
except ImportError:
    pass
from jax import jit as jjit
from jax import numpy as jnp

from .. import diffmahpop_params as dpp
from .. import mc_diffmahpop_kernels as mdk

N_T_PER_BIN = 5
LGSMAH_MIN = -15


@jjit
def mc_diffmah_preds(diffmahpop_u_params, pred_data):
    diffmahpop_params = dpp.get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    tarr, lgm_obs, t_obs, ran_key, lgt0 = pred_data
    _res = mdk._mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    ftpt0 = _res[3]
    dmhdt_tpt0 = _res[5]
    log_mah_tpt0 = _res[6]
    dmhdt_tp = _res[7]
    log_mah_tp = _res[8]

    dmhdt_tpt0 = jnp.clip(dmhdt_tpt0, 10**LGSMAH_MIN)  # make log-safe
    dmhdt_tp = jnp.clip(dmhdt_tp, 10**LGSMAH_MIN)  # make log-safe

    lgsmah_tpt0 = jnp.log10(dmhdt_tpt0) - log_mah_tpt0  # compute lgsmah
    lgsmah_tpt0 = jnp.clip(lgsmah_tpt0, LGSMAH_MIN)  # impose lgsMAH clip

    lgsmah_tp = jnp.log10(dmhdt_tp) - log_mah_tpt0  # compute lgsmah
    lgsmah_tp = jnp.clip(lgsmah_tp, LGSMAH_MIN)  # impose lgsMAH clip

    return lgsmah_tpt0, log_mah_tpt0, lgsmah_tp, log_mah_tp, ftpt0


@jjit
def get_single_sample_self_fit_target_data(
    u_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    pred_data = tarr, lgm_obs, t_obs, ran_key, lgt0
    _res = mc_diffmah_preds(u_params, pred_data)
    lgsmah_tpt0, log_mah_tpt0, lgsmah_tp, log_mah_tp, ftpt0 = _res
    weights_target = jnp.concatenate((ftpt0, 1 - ftpt0))
    lgsmah_target = jnp.concatenate((lgsmah_tpt0, lgsmah_tp))
    log_mahs_target = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    X_target = jnp.array((lgsmah_target, log_mahs_target)).swapaxes(0, 1)
    return X_target, weights_target


@jjit
def get_single_cen_sample_target_data(mah_params, t_peak, tarr, lgm_obs, t_obs, lgt0):
    dmhdt, log_mah = mah_halopop(mah_params, tarr, t_peak, lgt0)

    # renormalize MAHs to zero to at lgm_obs
    delta_log_mahs_target = log_mah - lgm_obs

    frac_peaked_target = jnp.mean(dmhdt == 0, axis=0)
    weights_target = jnp.where(dmhdt == 0, 0.0, 1.0)

    log_dmhdt = jnp.log10(jnp.clip(dmhdt, 10**LGSMAH_MIN))

    lgsmah_target = log_dmhdt - log_mah  # use log_mah since dmhdt was never rescaled
    lgsmah_target = jnp.clip(lgsmah_target, LGSMAH_MIN)

    X_target = jnp.array((lgsmah_target, delta_log_mahs_target)).swapaxes(0, 1)

    return X_target, weights_target, frac_peaked_target


@jjit
def single_sample_kde_loss_self_fit(
    diffmahpop_u_params,
    tarr,
    lgm_obs,
    t_obs,
    ran_key,
    lgt0,
    X_target,
    weights_target,
):
    kcalc0 = kdescent.KCalc(X_target[:, :, 0], weights_target)
    kcalc1 = kdescent.KCalc(X_target[:, :, 1], weights_target)
    kcalc2 = kdescent.KCalc(X_target[:, :, 2], weights_target)
    kcalc3 = kdescent.KCalc(X_target[:, :, 3], weights_target)
    kcalc4 = kdescent.KCalc(X_target[:, :, 4], weights_target)

    ran_key, pred_key = jran.split(ran_key, 2)
    pred_data = tarr, lgm_obs, t_obs, pred_key, lgt0
    _res = mc_diffmah_preds(diffmahpop_u_params, pred_data)
    dmhdt_tpt0, log_mah_tpt0, dmhdt_tp, log_mah_tp, ftpt0 = _res

    weights_pred = jnp.concatenate((ftpt0, 1 - ftpt0))
    dmhdts_pred = jnp.concatenate((dmhdt_tpt0, dmhdt_tp))
    log_mahs_pred = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    X_preds = jnp.array((dmhdts_pred, log_mahs_pred)).swapaxes(0, 1)

    kcalc_keys = jran.split(ran_key, N_T_PER_BIN)

    model_counts0, truth_counts0 = kcalc0.compare_kde_counts(
        kcalc_keys[0], X_preds[:, :, 0], weights_pred
    )
    model_counts1, truth_counts1 = kcalc1.compare_kde_counts(
        kcalc_keys[1], X_preds[:, :, 1], weights_pred
    )
    model_counts2, truth_counts2 = kcalc2.compare_kde_counts(
        kcalc_keys[2], X_preds[:, :, 2], weights_pred
    )
    model_counts3, truth_counts3 = kcalc3.compare_kde_counts(
        kcalc_keys[3], X_preds[:, :, 3], weights_pred
    )
    model_counts4, truth_counts4 = kcalc4.compare_kde_counts(
        kcalc_keys[4], X_preds[:, :, 4], weights_pred
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


def single_sample_kde_loss_kern(
    diffmahpop_u_params,
    tarr,
    lgm_obs,
    t_obs,
    ran_key,
    lgt0,
    X_target,
    weights_target,
    frac_peaked_target,
):
    kcalc0 = kdescent.KCalc(X_target[:, :, 0], weights_target[:, 0])
    kcalc1 = kdescent.KCalc(X_target[:, :, 1], weights_target[:, 1])
    kcalc2 = kdescent.KCalc(X_target[:, :, 2], weights_target[:, 2])
    kcalc3 = kdescent.KCalc(X_target[:, :, 3], weights_target[:, 3])

    lgsmah_target_t_obs = X_target[:, 0, 4].reshape((-1, 1))
    kcalc_t_obs = kdescent.KCalc(lgsmah_target_t_obs)

    ran_key, pred_key = jran.split(ran_key, 2)
    pred_data = tarr, lgm_obs, t_obs, pred_key, lgt0
    _res = mc_diffmah_preds(diffmahpop_u_params, pred_data)
    lgsmah_tpt0, log_mah_tpt0, lgsmah_tp, log_mah_tp, ftpt0 = _res

    weights_ftpt0 = jnp.concatenate((ftpt0, 1 - ftpt0))
    lgsmah_pred = jnp.concatenate((lgsmah_tpt0, lgsmah_tp))
    log_mahs_pred = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    delta_log_mahs_pred = log_mahs_pred - lgm_obs

    frac_peaked_pred = jnp.average(
        lgsmah_pred == LGSMAH_MIN, axis=0, weights=weights_ftpt0
    )

    weights_tp = jnp.where(lgsmah_tp == LGSMAH_MIN, 0.0, 1.0)
    weights_tpt0 = jnp.ones_like(weights_tp)
    weights = jnp.concatenate((weights_tpt0, weights_tpt0))
    weights = weights * weights_ftpt0.reshape((-1, 1))

    X_preds = jnp.array((lgsmah_pred, delta_log_mahs_pred)).swapaxes(0, 1)

    kcalc_keys = jran.split(ran_key, N_T_PER_BIN)

    model_counts0, truth_counts0 = kcalc0.compare_kde_counts(
        kcalc_keys[0], X_preds[:, :, 0], weights[:, 0]
    )
    model_counts1, truth_counts1 = kcalc1.compare_kde_counts(
        kcalc_keys[1], X_preds[:, :, 1], weights[:, 1]
    )
    model_counts2, truth_counts2 = kcalc2.compare_kde_counts(
        kcalc_keys[2], X_preds[:, :, 2], weights[:, 2]
    )
    model_counts3, truth_counts3 = kcalc3.compare_kde_counts(
        kcalc_keys[3], X_preds[:, :, 3], weights[:, 3]
    )

    lgsmah_pred_t_obs = lgsmah_pred[:, -1].reshape((-1, 1))
    model_counts4, truth_counts4 = kcalc_t_obs.compare_kde_counts(
        kcalc_keys[4], lgsmah_pred_t_obs, weights[:, 4]
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

    frac_peaked_diff = frac_peaked_pred - frac_peaked_target
    loss_frac_peaked = jnp.mean(frac_peaked_diff**2)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss_frac_peaked
    return (
        loss0,
        loss1,
        loss2,
        loss3,
        loss4,
        loss_frac_peaked,
    )


single_sample_kde_loss_and_grad_self_fit = jjit(
    value_and_grad(single_sample_kde_loss_self_fit)
)

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
    X_targets,
    weights_targets,
):
    losses = _multisample_kde_loss_self_fit(
        diffmahpop_u_params,
        tarr_matrix,
        lgmobsarr,
        tobsarr,
        ran_keys,
        lgt0,
        X_targets,
        weights_targets,
    )
    loss = jnp.mean(losses)
    return loss


multisample_kde_loss_and_grad_self_fit = jjit(
    value_and_grad(multisample_kde_loss_self_fit)
)
