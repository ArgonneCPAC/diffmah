"""
"""

from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..diffmah_kernels import (
    DiffmahParams,
    DiffmahUParams,
    get_bounded_mah_params,
    get_unbounded_mah_params,
    mah_singlehalo,
)
from .bimod_censat_params import get_component_model_params
from .bimod_logm0_kernels.logm0_pop_bimod import (
    _pred_logm0_kern_early,
    _pred_logm0_kern_late,
)
from .covariance_kernels import _get_diffmahpop_cov
from .early_index_pop import _pred_early_index_kern
from .late_index_pop import _pred_late_index_kern
from .logtc_pop import _pred_logtc_kern
from .t_peak_kernels.tp_pdf_monocens import mc_tpeak_singlecen

NH_PER_M0BIN = 200


@jjit
def _mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    _early = _mean_diffmah_params_early(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )
    mah_params_early, t_peak_early = _early

    _late = _mean_diffmah_params_late(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    mah_params_late, t_peak_late = _late
    return mah_params_early, t_peak_early, mah_params_late, t_peak_late


@jjit
def _mean_diffmah_params_early(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    t_0 = 10**lgt0
    model_params = get_component_model_params(diffmahpop_params)
    (
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    t_peak = mc_tpeak_singlecen(tp_pdf_cens_params, lgm_obs, tpc_key, t_0)

    logm0 = _pred_logm0_kern_early(logm0_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_kern(late_index_params, lgm_obs)

    mah_params = DiffmahParams(logm0, logtc, early_index, late_index)

    return mah_params, t_peak


@jjit
def _mean_diffmah_params_late(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    t_0 = 10**lgt0
    model_params = get_component_model_params(diffmahpop_params)
    (
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    t_peak = mc_tpeak_singlecen(tp_pdf_cens_params, lgm_obs, tpc_key, t_0)

    logm0 = _pred_logm0_kern_late(logm0_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_kern(late_index_params, lgm_obs)

    mah_params = DiffmahParams(logm0, logtc, early_index, late_index)

    return mah_params, t_peak


@jjit
def mc_diffmah_params_singlecen(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    _res = _mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    mean_mah_params_early, t_peak_early, mean_mah_params_late, t_peak_late = _res

    mean_mah_u_params_early = get_unbounded_mah_params(mean_mah_params_early)
    mean_mah_u_params_late = get_unbounded_mah_params(mean_mah_params_late)

    cov = _get_diffmahpop_cov(diffmahpop_params, lgm_obs)

    early_key, late_key = jran.split(ran_key, 2)
    ran_diffmah_u_params_tp_early = jran.multivariate_normal(
        early_key, jnp.array(mean_mah_u_params_early), cov, shape=()
    )
    ran_diffmah_u_params_early = DiffmahUParams(*ran_diffmah_u_params_tp_early)

    ran_diffmah_u_params_tp_late = jran.multivariate_normal(
        late_key, jnp.array(mean_mah_u_params_late), cov, shape=()
    )
    ran_diffmah_u_params_late = DiffmahUParams(*ran_diffmah_u_params_tp_late)

    mah_params_early = get_bounded_mah_params(ran_diffmah_u_params_early)
    mah_params_late = get_bounded_mah_params(ran_diffmah_u_params_late)

    return mah_params_early, t_peak_early, mah_params_late, t_peak_late


@jjit
def _mc_diffmah_singlecen(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    _res = mc_diffmah_params_singlecen(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    mah_params_early, t_peak_early, mah_params_late, t_peak_late = _res
    dmhdt_early, log_mah_early = mah_singlehalo(
        mah_params_early, tarr, t_peak_early, lgt0
    )
    dmhdt_late, log_mah_late = mah_singlehalo(mah_params_late, tarr, t_peak_late, lgt0)

    _ret_early = (mah_params_early, t_peak_early, dmhdt_early, log_mah_early)
    _ret_late = (mah_params_late, t_peak_late, dmhdt_late, log_mah_late)
    _ret = (*_ret_early, *_ret_late)
    return _ret


_V = (None, None, 0, 0, 0, None)
_mc_diffmah_singlecen_vmap_kern = jjit(vmap(_mc_diffmah_singlecen, in_axes=_V))


@partial(jjit, static_argnames=["n_mc"])
def _mc_diffmah_halo_sample(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, n_mc=NH_PER_M0BIN
):
    zz = jnp.zeros(n_mc)
    ran_keys = jran.split(ran_key, n_mc)
    return _mc_diffmah_singlecen_vmap_kern(
        diffmahpop_params, tarr, lgm_obs + zz, t_obs + zz, ran_keys, lgt0
    )


@jjit
def predict_mah_moments_singlebin(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    _res = _mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    _res_early = _res[:4]
    _res_late = _res[4:]
    dmhdt_early, log_mah_early = _res_early[2:]
    dmhdt_late, log_mah_late = _res_late[2:]

    n_early = log_mah_early.shape[0]
    n_late = log_mah_early.shape[0]

    frac_early = 0.5
    frac_late = 1.0 - frac_early
    weights_early = 1 / float(n_early)
    weights_late = 1 / float(n_late)
    w_e = frac_late * weights_late
    w_l = (1 - frac_late) * weights_early

    mean_log_mah = jnp.sum(log_mah_early * w_e + log_mah_late + w_l, axis=0)

    dlgm_sq_early = (log_mah_early - mean_log_mah) ** 2
    dlgm_sq_late = (log_mah_late - mean_log_mah) ** 2
    var_log_mah = jnp.sum(dlgm_sq_early * w_e + dlgm_sq_late + w_l, axis=0)
    std_log_mah = jnp.sqrt(var_log_mah)

    frac_peaked = 0.5 + jnp.zeros_like(mean_log_mah)

    return mean_log_mah, std_log_mah, frac_peaked
