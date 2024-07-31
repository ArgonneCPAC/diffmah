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
from .t_peak_kernels.tp_pdf_cens import mc_tpeak_singlecen
from .t_peak_kernels.tp_pdf_sats import mc_tpeak_singlesat

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5
NH_PER_M0BIN = 200


@jjit
def mc_mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    t_0 = 10**lgt0
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
    frac_tpt0_cens = ftpt0_cens._ftpt0_kernel(ftpt0_cens_params, lgm_obs, t_obs)

    tpc_key, tps_key, ran_key = jran.split(ran_key, 3)

    args = tp_pdf_cens_params, tpc_key, lgm_obs, t_obs, t_0
    t_peak_cens = mc_tpeak_singlecen(*args)

    t_peak_sats = mc_tpeak_singlesat(tp_pdf_sats_params, ran_key, lgm_obs, t_obs)

    ftpt0_key, ran_key = jran.split(ran_key, 2)
    mc_tpt0_cens = jran.uniform(ftpt0_key, shape=()) < frac_tpt0_cens

    logm0_tpt0_cens = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_0)
    logm0_tp_cens = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak_cens)
    logm0_sats = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak_sats)

    logtc_tpt0_cens = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_0)
    logtc_tp_cens = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak_cens)
    logtc_sats = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak_sats)

    early_index_tpt0_cens = _pred_early_index_kern(
        early_index_params, lgm_obs, t_obs, t_0
    )
    early_index_tp_cens = _pred_early_index_kern(
        early_index_params, lgm_obs, t_obs, t_peak_cens
    )
    early_index_sats = _pred_early_index_kern(
        early_index_params, lgm_obs, t_obs, t_peak_sats
    )

    late_index_tpt0_cens = _pred_late_index_kern(late_index_params, lgm_obs)
    late_index_tp_cens = _pred_late_index_kern(late_index_params, lgm_obs)
    late_index_sats = _pred_late_index_kern(late_index_params, lgm_obs)

    dmah_tpt0_cens = DiffmahParams(
        logm0_tpt0_cens, logtc_tpt0_cens, early_index_tpt0_cens, late_index_tpt0_cens
    )
    dmah_tp_cens = DiffmahParams(
        logm0_tp_cens, logtc_tp_cens, early_index_tp_cens, late_index_tp_cens
    )

    dmah_sats = DiffmahParams(logm0_sats, logtc_sats, early_index_sats, late_index_sats)

    return (
        dmah_tpt0_cens,
        dmah_tp_cens,
        t_peak_cens,
        frac_tpt0_cens,
        mc_tpt0_cens,
        t_peak_sats,
        dmah_sats,
    )


@jjit
def mc_diffmah_params_single_censat(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    _res = mc_mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    (
        dmah_tpt0_cens,
        dmah_tp_cens,
        t_peak_cens,
        frac_tpt0_cens,
        mc_tpt0_cens,
        t_peak_sats,
        dmah_sats,
    ) = _res
    u_dmah_tpt0_cens = get_unbounded_mah_params(dmah_tpt0_cens)
    u_dmah_tp_cens = get_unbounded_mah_params(dmah_sats)

    u_dmah_sats = get_unbounded_mah_params(dmah_tp_cens)

    cov = _get_diffmahpop_cov(diffmahpop_params, lgm_obs)

    ran_key, tpt0_cens_key, tp_cens_key = jran.split(ran_key, 3)
    ran_diffmah_u_params_tpt0_cens = jran.multivariate_normal(
        tpt0_cens_key, jnp.array(u_dmah_tpt0_cens), cov, shape=()
    )
    ran_diffmah_u_params_tp_cens = jran.multivariate_normal(
        tp_cens_key, jnp.array(u_dmah_tp_cens), cov, shape=()
    )
    sats_key = tp_cens_key
    ran_diffmah_u_params_sats = jran.multivariate_normal(
        sats_key, jnp.array(u_dmah_sats), cov, shape=()
    )

    ran_diffmah_u_params_tpt0_cens = DiffmahUParams(*ran_diffmah_u_params_tpt0_cens)
    ran_diffmah_u_params_tp_cens = DiffmahUParams(*ran_diffmah_u_params_tp_cens)
    ran_diffmah_u_params_sats = DiffmahUParams(*ran_diffmah_u_params_sats)

    mah_params_tpt0_cens = get_bounded_mah_params(ran_diffmah_u_params_tpt0_cens)
    mah_params_tp_cens = get_bounded_mah_params(ran_diffmah_u_params_tp_cens)
    mah_params_sats = get_bounded_mah_params(ran_diffmah_u_params_sats)

    return (
        mah_params_tpt0_cens,
        mah_params_tp_cens,
        t_peak_cens,
        frac_tpt0_cens,
        mc_tpt0_cens,
        t_peak_sats,
        mah_params_sats,
    )


_A = (None, 0, 0, 0, None)
_mc_diffmah_params_vmap_kern = jjit(vmap(mc_diffmah_params_single_censat, in_axes=_A))


@jjit
def mc_diffmah_params_cenpop(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    ran_keys = jran.split(ran_key, lgm_obs.size)
    return _mc_diffmah_params_vmap_kern(
        diffmahpop_params, lgm_obs, t_obs, ran_keys, lgt0
    )


@jjit
def _mc_diffmah_single_censat(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    _res = mc_diffmah_params_single_censat(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )
    (
        mah_params_tpt0_cens,
        mah_params_tp_cens,
        t_peak_cens,
        frac_tpt0_cens,
        mc_tpt0_cens,
        t_peak_sats,
        mah_params_sats,
    ) = _res
    dmhdt_tpt0_cens, log_mah_tpt0_cens = mah_singlehalo(
        mah_params_tpt0_cens, tarr, 10**lgt0, lgt0
    )
    dmhdt_tp_cens, log_mah_tp_cens = mah_singlehalo(
        mah_params_tp_cens, tarr, t_peak_cens, lgt0
    )
    dmhdt_sats, log_mah_sats = mah_singlehalo(
        mah_params_tp_cens, tarr, t_peak_sats, lgt0
    )
    _ret = (
        mah_params_tpt0_cens,
        mah_params_tp_cens,
        t_peak_cens,
        frac_tpt0_cens,
        mc_tpt0_cens,
        dmhdt_tpt0_cens,
        log_mah_tpt0_cens,
        dmhdt_tp_cens,
        log_mah_tp_cens,
        dmhdt_sats,
        log_mah_sats,
    )
    return _ret


_V = (None, None, 0, 0, 0, None)
_mc_diffmah_single_censat_vmap_kern = jjit(vmap(_mc_diffmah_single_censat, in_axes=_V))


@partial(jjit, static_argnames=["n_mc"])
def _mc_diffmah_halo_sample_censat(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0, n_mc=NH_PER_M0BIN
):
    zz = jnp.zeros(n_mc)
    ran_keys = jran.split(ran_key, n_mc)
    return _mc_diffmah_single_censat_vmap_kern(
        diffmahpop_params, tarr, lgm_obs + zz, t_obs + zz, ran_keys, lgt0
    )


@jjit
def predict_mah_moments_singlebin_censat(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    _res = _mc_diffmah_halo_sample_censat(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    (
        mah_params_tpt0_cens,
        mah_params_tp_cens,
        t_peak_cens,
        frac_tpt0_cens,
        mc_tpt0_cens,
        dmhdt_tpt0_cens,
        log_mah_tpt0_cens,
        dmhdt_tp_cens,
        log_mah_tp_cens,
        dmhdt_sats,
        log_mah_sats,
    ) = _res

    f = frac_tpt0_cens.reshape((-1, 1))
    mean_log_mah_cens = jnp.mean(
        f * log_mah_tpt0_cens + (1 - f) * log_mah_tp_cens, axis=0
    )
    std_log_mah_cens = jnp.std(
        f * log_mah_tpt0_cens + (1 - f) * log_mah_tp_cens, axis=0
    )

    mean_log_mah_sats = jnp.mean(f * log_mah_sats + (1 - f) * log_mah_sats, axis=0)
    std_log_mah_sats = jnp.std(f * log_mah_sats + (1 - f) * log_mah_sats, axis=0)

    return mean_log_mah_cens, std_log_mah_cens, mean_log_mah_sats, std_log_mah_sats
