"""
"""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..diffmah_kernels import (
    DEFAULT_MAH_PARAMS,
    DiffmahParams,
    DiffmahUParams,
    get_bounded_mah_params,
    get_unbounded_mah_params,
    mah_singlehalo,
)
from .bimod_censat_params import get_component_model_params
from .bimod_logm0_sats.logm0_pop_bimod_sats import (
    _pred_logm0_kern_early,
    _pred_logm0_kern_late,
)
from .covariance_kernels import _get_diffmahpop_cov
from .early_index_bimod import _pred_early_index_early, _pred_early_index_late
from .frac_early_cens import _frac_early_cens_kern
from .late_index_bimod import _pred_late_index_early, _pred_late_index_late
from .logtc_early import _pred_logtc_kern as _pred_logtc_early
from .logtc_late import _pred_logtc_kern as _pred_logtc_late
from .t_peak_kernels.tp_pdf_sats import mc_tpeak_singlesat

NH_PER_M0BIN = 200


@jjit
def _mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    mah_params_early = _mean_diffmah_params_early(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )

    mah_params_late = _mean_diffmah_params_late(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )

    frac_early_cens = _frac_early_cens_kern(diffmahpop_params, lgm_obs, t_obs)

    return mah_params_early, mah_params_late, frac_early_cens


@jjit
def _mean_diffmah_params_t_peak(diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key):
    mah_params_early = _mean_diffmah_params_early_t_peak(
        diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key
    )

    mah_params_late = _mean_diffmah_params_late_t_peak(
        diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key
    )

    frac_early_cens = _frac_early_cens_kern(diffmahpop_params, lgm_obs, t_obs)

    return mah_params_early, mah_params_late, frac_early_cens


@jjit
def _mean_diffmah_params_early(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    model_params = get_component_model_params(diffmahpop_params)
    (
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logm0_sats_params,
        logtc_params,
        early_index_params,
        late_index_params,
        fec_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    args = tp_pdf_sats_params, tpc_key, lgm_obs, t_obs
    t_peak = mc_tpeak_singlesat(*args)

    logm0 = _pred_logm0_kern_early(logm0_sats_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_early(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_early(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_early(late_index_params, lgm_obs)

    mah_params = DiffmahParams(logm0, logtc, early_index, late_index, t_peak)

    return mah_params


@jjit
def _mean_diffmah_params_early_t_peak(
    diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key
):
    model_params = get_component_model_params(diffmahpop_params)
    (
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logm0_sats_params,
        logtc_params,
        early_index_params,
        late_index_params,
        fec_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    logm0 = _pred_logm0_kern_early(logm0_sats_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_early(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_early(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_early(late_index_params, lgm_obs)

    mah_params = DiffmahParams(logm0, logtc, early_index, late_index, t_peak)

    return mah_params


@jjit
def _mean_diffmah_params_late(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    model_params = get_component_model_params(diffmahpop_params)
    (
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logm0_sats_params,
        logtc_params,
        early_index_params,
        late_index_params,
        fec_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    args = tp_pdf_sats_params, tpc_key, lgm_obs, t_obs
    t_peak = mc_tpeak_singlesat(*args)

    logm0 = _pred_logm0_kern_late(logm0_sats_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_late(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_late(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_late(late_index_params, lgm_obs)

    mah_params = DiffmahParams(logm0, logtc, early_index, late_index, t_peak)

    return mah_params


@jjit
def _mean_diffmah_params_late_t_peak(
    diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key
):
    model_params = get_component_model_params(diffmahpop_params)
    (
        tp_pdf_cens_params,
        tp_pdf_sats_params,
        logm0_params,
        logm0_sats_params,
        logtc_params,
        early_index_params,
        late_index_params,
        fec_params,
        cov_params,
    ) = model_params

    tpc_key, ran_key = jran.split(ran_key, 2)

    logm0 = _pred_logm0_kern_late(logm0_sats_params, lgm_obs, t_obs, t_peak)
    logtc = _pred_logtc_late(logtc_params, lgm_obs, t_obs, t_peak)
    early_index = _pred_early_index_late(early_index_params, lgm_obs, t_obs, t_peak)
    late_index = _pred_late_index_late(late_index_params, lgm_obs)

    mah_params = DiffmahParams(logm0, logtc, early_index, late_index, t_peak)

    return mah_params


@jjit
def mc_diffmah_params_singlesat(
    diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0, t_peak=None
):
    if t_peak is None:
        _res = _mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    else:
        _res = _mean_diffmah_params_t_peak(
            diffmahpop_params, lgm_obs, t_obs, t_peak, ran_key
        )

    mean_mah_params_early, mean_mah_params_late, frac_early_cens = _res

    mean_mah_u_params_early = get_unbounded_mah_params(mean_mah_params_early)
    mean_mah_u_params_late = get_unbounded_mah_params(mean_mah_params_late)

    cov = _get_diffmahpop_cov(diffmahpop_params, lgm_obs)

    early_key, late_key = jran.split(ran_key, 2)
    ran_diffmah_u_params_tp_early = jran.multivariate_normal(
        early_key, jnp.array(mean_mah_u_params_early)[:-1], cov, shape=()
    )
    ran_diffmah_u_params_early = DiffmahUParams(
        *ran_diffmah_u_params_tp_early, mean_mah_u_params_early[-1]
    )

    ran_diffmah_u_params_tp_late = jran.multivariate_normal(
        late_key, jnp.array(mean_mah_u_params_late)[:-1], cov, shape=()
    )
    ran_diffmah_u_params_late = DiffmahUParams(
        *ran_diffmah_u_params_tp_late, mean_mah_u_params_late[-1]
    )

    mah_params_early = get_bounded_mah_params(ran_diffmah_u_params_early)
    mah_params_late = get_bounded_mah_params(ran_diffmah_u_params_late)

    return mah_params_early, mah_params_late, frac_early_cens


@jjit
def _mc_diffmah_singlesat(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    _res = mc_diffmah_params_singlesat(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0)
    mah_params_early, mah_params_late, frac_early_cens = _res
    dmhdt_early, log_mah_early = mah_singlehalo(mah_params_early, tarr, lgt0)
    dmhdt_late, log_mah_late = mah_singlehalo(mah_params_late, tarr, lgt0)

    _ret_early = (mah_params_early, dmhdt_early, log_mah_early)
    _ret_late = (mah_params_late, dmhdt_late, log_mah_late)
    _ret = (*_ret_early, *_ret_late, frac_early_cens)
    return _ret


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


_P1 = (None, 0, 0, 0, None, None)
mc_diffmah_params_satpop_kern1 = jjit(vmap(mc_diffmah_params_singlesat, in_axes=_P1))

_P2 = (None, 0, 0, 0, None, 0)
mc_diffmah_params_satpop_kern2 = jjit(vmap(mc_diffmah_params_singlesat, in_axes=_P2))


@jjit
def mc_diffmah_satpop(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0, t_peak=None):
    """"""
    early_late_key, ran_key = jran.split(ran_key, 2)
    ran_keys = jran.split(ran_key, lgm_obs.size)
    args = diffmahpop_params, lgm_obs, t_obs, ran_keys, lgt0, t_peak
    if t_peak is None:
        _res = mc_diffmah_params_satpop_kern1(*args)
    else:
        _res = mc_diffmah_params_satpop_kern2(*args)
    mah_params_early, mah_params_late, frac_early_cens = _res
    uran = jran.uniform(early_late_key, shape=frac_early_cens.shape)
    mc_early = uran < frac_early_cens
    _p = [jnp.where(mc_early, x, y) for x, y in zip(mah_params_early, mah_params_late)]
    mah_params = DEFAULT_MAH_PARAMS._make(_p)

    halopop_keys = (
        "mah_params",
        "mah_params_early",
        "mah_params_late",
        "frac_early_cens",
        "mc_early",
    )
    HaloPop = namedtuple("HaloPop", halopop_keys)
    return HaloPop(
        mah_params, mah_params_early, mah_params_late, frac_early_cens, mc_early
    )


@jjit
def mc_satpop(diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0):
    """"""
    n_mc = lgm_obs.shape[0]
    ran_keys = jran.split(ran_key, n_mc + 1)
    dmah_keys = ran_keys[:-1]
    uran_key = ran_keys[-1]
    _res = _mc_diffmah_singlesat_vmap_kern(
        diffmahpop_params, tarr, lgm_obs, t_obs, dmah_keys, lgt0
    )
    p_e, dmhdt_early, log_mah_early = _res[0:3]
    p_l, dmhdt_late, log_mah_late = _res[3:6]
    frac_early_cens = _res[6]

    uran = jran.uniform(uran_key, minval=0, maxval=1, shape=lgm_obs.shape)
    msk = uran < frac_early_cens

    pns = DEFAULT_MAH_PARAMS._fields
    mah_params = [jnp.where(msk, getattr(p_e, pn), getattr(p_l, pn)) for pn in pns]
    mah_params = DEFAULT_MAH_PARAMS._make(mah_params)

    dmhdt = jnp.where(msk.reshape((-1, 1)), dmhdt_early, dmhdt_late)
    log_mah = jnp.where(msk.reshape((-1, 1)), log_mah_early, log_mah_late)

    halopop_keys = ("mah_params", "dmhdt", "log_mah")
    HaloPop = namedtuple("HaloPop", halopop_keys)
    return HaloPop(mah_params, dmhdt, log_mah)


@jjit
def predict_mah_moments_singlebin(
    diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
):
    _res = _mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    _res_early = _res[:3]
    _res_late = _res[3:6]
    frac_early = _res[6]
    mah_params_early, dmhdt_early, log_mah_early = _res_early
    mah_params_late, dmhdt_late, log_mah_late = _res_late

    n_early = log_mah_early.shape[0]
    n_late = log_mah_late.shape[0]

    frac_late = 1.0 - frac_early
    weights_early = 1 / float(n_early)
    weights_late = 1 / float(n_late)
    w_e = (frac_early * weights_late).reshape((-1, 1))
    w_l = (frac_late * weights_early).reshape((-1, 1))

    mu_e = jnp.sum(log_mah_early * w_e, axis=0)
    mu_l = jnp.sum(log_mah_late * w_l, axis=0)
    mean_log_mah = mu_e + mu_l

    dlgm_sq_early = (log_mah_early - mean_log_mah) ** 2
    dlgm_sq_late = (log_mah_late - mean_log_mah) ** 2
    var_e = jnp.sum(dlgm_sq_early * w_e, axis=0)
    var_l = jnp.sum(dlgm_sq_late * w_l, axis=0)
    var_log_mah = var_e + var_l
    std_log_mah = jnp.sqrt(var_log_mah)

    frac_peaked = 0.5 + jnp.zeros_like(mean_log_mah)

    return mean_log_mah, std_log_mah, frac_peaked
