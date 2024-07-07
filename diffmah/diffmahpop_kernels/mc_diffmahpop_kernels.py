"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from ..diffmah_kernels import DiffmahParams, mah_halopop, mah_singlehalo
from . import ftpt0_cens
from .diffmahpop_params import (
    DEFAULT_DIFFMAHPOP_U_PARAMS,
    get_component_model_params,
    get_diffmahpop_params_from_u_params,
)
from .early_index_pop import _pred_early_index_kern
from .late_index_pop import _pred_late_index_kern
from .logm0_kernels.logm0_pop import _pred_logm0_kern
from .logtc_pop import _pred_logtc_kern
from .t_peak_kernels.tp_pdf_cens import mc_tpeak_cens

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5


@jjit
def mc_tp_avg_dmah_params_singlecen(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    t_0 = 10**lgt0
    model_params = get_component_model_params(diffmahpop_params)
    (
        ftpt0_cens_params,
        tp_pdf_cens_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
    ) = model_params
    ftpt0 = ftpt0_cens._ftpt0_kernel(ftpt0_cens_params, lgm_obs, t_obs)

    tpc_key, ran_key = jran.split(ran_key, 2)
    ZZ = jnp.zeros(N_TP_PER_HALO)

    lgm_obs = lgm_obs + ZZ
    t_obs = t_obs + ZZ
    args = tp_pdf_cens_params, tpc_key, lgm_obs, t_obs, t_0
    t_peak = mc_tpeak_cens(*args)

    ftpt0_key, ran_key = jran.split(ran_key, 2)
    mc_tpt0 = jran.uniform(ftpt0_key, shape=(N_TP_PER_HALO,)) < ftpt0

    logm0_tpt0 = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_0 + ZZ)
    logm0_tp = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)

    logtc_tpt0 = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_0 + ZZ)
    logtc_tp = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak)

    early_index_tpt0 = _pred_early_index_kern(
        early_index_params, lgm_obs, t_obs, t_0 + ZZ
    )
    early_index_tp = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_peak)

    late_index_tpt0 = _pred_late_index_kern(late_index_params, lgm_obs)
    late_index_tp = _pred_late_index_kern(late_index_params, lgm_obs)

    dmah_tpt0 = DiffmahParams(logm0_tpt0, logtc_tpt0, early_index_tpt0, late_index_tpt0)
    dmah_tp = DiffmahParams(logm0_tp, logtc_tp, early_index_tp, late_index_tp)

    return dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0


@jjit
def mc_tp_avg_mah_singlecen(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    dmah_tpt0, dmah_tp, t_peak, ftpt0, __ = mc_tp_avg_dmah_params_singlecen(
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
def _loss_kern_singlehalo(diffmahpop_params, loss_data):
    tarr, lgm_obs, t_obs, ran_key, lgt0, avg_log_mah_target = loss_data
    avg_log_mah_pred = mc_tp_avg_mah_singlecen(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    loss = _mse(avg_log_mah_pred, avg_log_mah_target)
    return loss


@jjit
def _loss_kern_singlehalo_u_params(diffmahpop_u_params, loss_data):
    diffmahpop_u_params = DEFAULT_DIFFMAHPOP_U_PARAMS._make(diffmahpop_u_params)
    diffmahpop_params = get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    return _loss_kern_singlehalo(diffmahpop_params, loss_data)


@jjit
def _loss_kern_singlehalo_subset_u_params(diffmahpop_subset_u_params, loss_data):

    diffmahpop_u_params = DEFAULT_DIFFMAHPOP_U_PARAMS._replace(
        **diffmahpop_subset_u_params._asdict()
    )
    diffmahpop_params = get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    return _loss_kern_singlehalo(diffmahpop_params, loss_data)


@jjit
def _loss_kern_multihalo_u_params(diffmahpop_u_params, loss_data_collection):
    loss = 0.0
    for loss_data in loss_data_collection:
        loss = loss + _loss_kern_singlehalo_u_params(diffmahpop_u_params, loss_data)
    return loss


loss_and_grads_multihalo = jjit(value_and_grad(_loss_kern_multihalo_u_params))


@jjit
def _loss_kern_multihalo_subset_u_params(
    diffmahpop_subset_u_params, loss_data_collection
):
    loss = 0.0
    for loss_data in loss_data_collection:
        loss = loss + _loss_kern_singlehalo_subset_u_params(
            diffmahpop_subset_u_params, loss_data
        )
    return loss


loss_and_grads_multihalo_subset = jjit(
    value_and_grad(_loss_kern_multihalo_subset_u_params)
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
