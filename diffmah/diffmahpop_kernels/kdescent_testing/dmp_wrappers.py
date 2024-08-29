"""
"""

from jax import value_and_grad, vmap

try:
    import kdescent
except ImportError:
    pass
from jax import jit as jjit
from jax import numpy as jnp

from .. import diffmahpop_params as dpp
from .. import mc_diffmahpop_kernels as mdk


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
    pred_data = tarr, lgm_obs, t_obs, ran_key, lgt0
    _res = mc_diffmah_preds(diffmahpop_u_params, pred_data)
    log_mah_tpt0, log_mah_tp, ftpt0 = _res
    log_mahs_pred = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    weights_pred = jnp.concatenate((ftpt0, 1 - ftpt0))

    kcalc = kdescent.KCalc(log_mahs_target, weights_target)
    model_counts, truth_counts = kcalc.compare_kde_counts(
        ran_key, log_mahs_pred, weights_pred
    )
    diff = model_counts - truth_counts
    loss = jnp.mean(diff**2)
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
