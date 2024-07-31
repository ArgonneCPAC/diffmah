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
    mah_halopop,
    mah_singlehalo,
)
from . import ftpt0_cens
from .covariance_kernels import _get_diffmahpop_cov
from .diffmahpop_params_censat import get_component_model_params
from .early_index_pop import _pred_early_index_kern
from .late_index_pop import _pred_late_index_kern
from .logm0_kernels.logm0_pop import _pred_logm0_kern
from .logtc_pop import _pred_logtc_kern
from .t_peak_kernels.tp_pdf_sats import mc_tpeak_singlesat

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5
NH_PER_M0BIN = 200


@jjit
def mc_mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    model_params = get_component_model_params(diffmahpop_params)
    (
        ftpt0_cens_params,
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    lgm_obs = lgm_obs
    t_obs = t_obs
    args = tp_pdf_sats_params, tpc_key, lgm_obs, t_obs
    t_peak_sats = mc_tpeak_singlesat(*args)

    logm0_tp = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak_sats)

    logtc_tp = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak_sats)

    early_index_tp = _pred_early_index_kern(
        early_index_params, lgm_obs, t_obs, t_peak_sats
    )

    late_index_tp = _pred_late_index_kern(late_index_params, lgm_obs)

    dmah_sats = DiffmahParams(logm0_tp, logtc_tp, early_index_tp, late_index_tp)

    return dmah_sats, t_peak_sats


@jjit
def mc_diffmah_params_singlesat(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    dmah_sats, t_peak_sats = mc_mean_diffmah_params(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )
    u_dmah_sats = get_unbounded_mah_params(dmah_sats)

    cov = _get_diffmahpop_cov(diffmahpop_params, lgm_obs)

    ran_key, tpt0_key, tp_key = jran.split(ran_key, 3)

    ran_diffmah_u_params_tp = jran.multivariate_normal(
        tp_key, jnp.array(u_dmah_sats), cov, shape=()
    )
    ran_diffmah_u_params = DiffmahUParams(*ran_diffmah_u_params_tp)

    mah_params = get_bounded_mah_params(ran_diffmah_u_params)
    return mah_params, t_peak_sats


_A = (None, 0, 0, 0, None)
_mc_diffmah_params_vmap_kern = jjit(vmap(mc_diffmah_params_singlesat, in_axes=_A))


@jjit
def mc_diffmah_params_satpop(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    ran_keys = jran.split(ran_key, lgm_obs.size)
    return _mc_diffmah_params_vmap_kern(
        diffmahpop_params, lgm_obs, t_obs, ran_keys, lgt0
    )


@jjit
def _mc_diffmah_singlesat(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    _res = mc_diffmah_params_singlesat(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    mah_params, t_peak_sats = _res
    dmhdt_sats, log_mah_sats = mah_singlehalo(mah_params, tarr, t_peak_sats, lgt0)
    return mah_params, t_peak_sats, dmhdt_sats, log_mah_sats


_V = (None, None, 0, 0, 0, None)
_mc_diffmah_singlesat_vmap_kern = jjit(vmap(_mc_diffmah_singlesat, in_axes=_V))


@partial(jjit, static_argnames=["n_mc"])
def _mc_diffmah_halo_sample(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, n_mc=NH_PER_M0BIN
):
    zz = jnp.zeros(n_mc)
    ran_keys = jran.split(ran_key, n_mc)
    return _mc_diffmah_singlesat_vmap_kern(
        diffmahpop_params, tarr, lgm_obs + zz, t_obs + zz, ran_keys, lgt0
    )


@jjit
def predict_mah_moments_singlebin(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    _res = _mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    mah_params, t_peak_sats, dmhdt, log_mah = _res

    mean_log_mah = jnp.mean(log_mah, axis=0)
    std_log_mah = jnp.std(log_mah, axis=0)
    frac_peaked = jnp.mean(dmhdt == 0, axis=0)

    return mean_log_mah, std_log_mah, frac_peaked
