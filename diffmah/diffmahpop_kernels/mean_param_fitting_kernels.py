"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from ..diffmah_kernels import DiffmahParams, mah_halopop, mah_singlehalo
from .diffmahpop_params import (
    DEFAULT_DIFFMAHPOP_U_PARAMS,
    get_diffmahpop_params_from_u_params,
)
from .mc_diffmahpop_kernels import mc_mean_diffmah_params

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5


@jjit
def mc_tp_avg_mah_singlecen(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    dmah_tpt0, dmah_tp, t_peak, ftpt0, __ = mc_mean_diffmah_params(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )
    ZZ = jnp.zeros_like(t_peak)
    tpt0 = ZZ + 10**lgt0
    __, log_mah_tpt0 = mah_halopop(dmah_tpt0, tarr, tpt0, lgt0)
    __, log_mah_tp = mah_halopop(dmah_tp, tarr, t_peak, lgt0)

    avg_log_mah_tpt0 = jnp.mean(log_mah_tpt0, axis=0)
    avg_log_mah_tp = jnp.mean(log_mah_tp, axis=0)
    avg_log_mah = ftpt0 * avg_log_mah_tpt0 + (1 - ftpt0) * avg_log_mah_tp
    return avg_log_mah


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_scalar_kern(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target
):
    avg_log_mah_pred = mc_tp_avg_mah_singlecen(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    loss = _mse(avg_log_mah_pred, avg_log_mah_target)
    return loss


_A = (None, 0, 0, 0, 0, None, 0)
_loss_vmap_kern = jjit(vmap(_loss_scalar_kern, in_axes=_A))


@jjit
def multiloss_vmap(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target
):
    losses = _loss_vmap_kern(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target
    )
    return jnp.sum(losses)


multiloss_and_grads_vmap = jjit(value_and_grad(multiloss_vmap))


@jjit
def _loss_scalar_kern_subset_u_params(
    diffmahpop_subset_u_params, tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target
):
    diffmahpop_u_params = DEFAULT_DIFFMAHPOP_U_PARAMS._replace(
        **diffmahpop_subset_u_params._asdict()
    )
    diffmahpop_params = get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    args = diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target
    return _loss_scalar_kern(*args)


_A = (None, 0, 0, 0, 0, None, 0)
_loss_vmap_kern_subset_u_params = jjit(
    vmap(_loss_scalar_kern_subset_u_params, in_axes=_A)
)


@jjit
def multiloss_vmap_subset_u_params(diffmahpop_subset_u_params, loss_data):
    tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target = loss_data
    losses = _loss_vmap_kern_subset_u_params(
        diffmahpop_subset_u_params,
        tarr,
        lgm_obs,
        t_obs,
        ran_key,
        lgt0,
        avg_log_mah_target,
    )
    return jnp.sum(losses)


multiloss_and_grads_vmap_subset_u_params = jjit(
    value_and_grad(multiloss_vmap_subset_u_params)
)


def get_loss_data_singlehalo(mah_data, ih, lgt0, nt=50):
    mah_params_ih = DiffmahParams(
        *[mah_data[key][ih] for key in ("logm0", "logtc", "early_index", "late_index")]
    )
    t_obs = mah_data["t_obs"][ih]
    t_target = jnp.linspace(T_OBS_FIT_MIN, t_obs, nt)
    args = mah_params_ih, t_target, mah_data["t_peak"][ih], lgt0
    avg_log_mah_target = mah_singlehalo(*args)[1]
    ran_key = jran.key(ih)
    loss_data = (
        t_target,
        mah_data["logmp_at_z"][ih],
        mah_data["t_obs"][ih],
        ran_key,
        lgt0,
        avg_log_mah_target,
    )
    return loss_data
