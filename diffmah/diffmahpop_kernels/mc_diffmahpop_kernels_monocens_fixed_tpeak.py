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
from .covariance_kernels import _get_diffmahpop_cov
from .diffmahpop_params_monocensat import get_component_model_params
from .early_index_pop import _pred_early_index_kern
from .late_index_pop import _pred_late_index_kern
from .logm0_kernels.logm0_pop import _pred_logm0_kern
from .logtc_pop import _pred_logtc_kern

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5
NH_PER_M0BIN = 200


@jjit
def mc_mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, t_peak):
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

    lgm_obs = lgm_obs
    t_obs = t_obs

    logm0 = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_kern(late_index_params, lgm_obs)
    mah_params = DiffmahParams(logm0, logtc, early_index, late_index)

    return mah_params


@jjit
def mc_diffmah_params_singlecen(diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key):
    mean_mah_params = mc_mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, t_peak)
    mean_mah_u_params = get_unbounded_mah_params(mean_mah_params)

    cov = _get_diffmahpop_cov(diffmahpop_params, lgm_obs)

    ran_key, p_key = jran.split(ran_key, 2)
    ran_diffmah_u_params_tp = jran.multivariate_normal(
        p_key, jnp.array(mean_mah_u_params), cov, shape=()
    )
    ran_diffmah_u_params = DiffmahUParams(*ran_diffmah_u_params_tp)

    mah_params = get_bounded_mah_params(ran_diffmah_u_params)
    return mah_params


_A = (None, 0, 0, 0, 0)
_mc_diffmah_params_vmap_kern = jjit(vmap(mc_diffmah_params_singlecen, in_axes=_A))


@jjit
def mc_diffmah_params_cenpop(diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key, lgt0):
    ran_keys = jran.split(ran_key, lgm_obs.size)
    return _mc_diffmah_params_vmap_kern(
        diffmahpop_params, lgm_obs, t_obs, ran_keys, t_peak, lgt0
    )


@jjit
def _mc_diffmah_singlecen(
    diffmahpop_params, tarr, lgm_obs, t_obs, t_peak, ran_key, lgt0
):
    mah_params = mc_diffmah_params_singlecen(
        diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key
    )
    dmhdt, log_mah = mah_singlehalo(mah_params, tarr, t_peak, lgt0)
    _ret = (mah_params, dmhdt, log_mah)
    return _ret


_V = (None, None, 0, 0, 0, 0, None)
_mc_diffmah_singlecen_vmap_kern = jjit(vmap(_mc_diffmah_singlecen, in_axes=_V))


@jjit
def _mc_diffmah_halo_sample(
    diffmahpop_params, tarr, lgm_obs, t_obs, t_peak_sample, ran_key, lgt0
):
    zz = jnp.zeros_like(t_peak_sample)
    ran_keys = jran.split(ran_key, zz.size)
    return _mc_diffmah_singlecen_vmap_kern(
        diffmahpop_params, tarr, lgm_obs + zz, t_obs + zz, t_peak_sample, ran_keys, lgt0
    )


@jjit
def predict_mah_moments_singlebin(
    diffmahpop_params, tarr, lgm_obs, t_obs, t_peak_sample, ran_key, lgt0
):
    _res = _mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, t_peak_sample, ran_key, lgt0
    )
    mah_params, dmhdt, log_mah = _res

    mean_log_mah = jnp.mean(log_mah, axis=0)
    std_log_mah = jnp.std(log_mah, axis=0)

    frac_peaked = jnp.mean(dmhdt == 0, axis=0)

    return mean_log_mah, std_log_mah, frac_peaked
