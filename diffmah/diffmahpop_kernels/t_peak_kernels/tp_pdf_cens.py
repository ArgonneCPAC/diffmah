"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from ...bfgs_wrapper import bfgs_adam_fallback
from ...utils import _inverse_sigmoid, _sigmoid
from . import utp_pdf_kernels as tpk

LOC_T_OBS_K = 1.0

DEFAULT_TPCENS_PDICT = OrderedDict(
    tpck_scale_boost_early=0.314,
    tpck_scale_mh_lo=0.469,
    tpck_scale_mh_hi=0.292,
)

TPCENS_PBOUNDS_PDICT = OrderedDict(
    tpck_scale_boost_early=(0.01, 0.9),
    tpck_scale_mh_lo=(0.1, 0.9),
    tpck_scale_mh_hi=(0.1, 0.9),
)

TPCK_LGM0, TPCK_K = 12.5, 2.0
DEFAULT_TRUNCNORM_LOC = 0.95

TPCens_Params = namedtuple("TPCens_Params", DEFAULT_TPCENS_PDICT.keys())
_TPCens_UPNAMES = ["u_" + key for key in TPCens_Params._fields]
TPCens_UParams = namedtuple("TPCens_UParams", _TPCens_UPNAMES)


DEFAULT_TPCENS_PARAMS = TPCens_Params(**DEFAULT_TPCENS_PDICT)
TPCENS_PBOUNDS = TPCens_Params(**TPCENS_PBOUNDS_PDICT)

K_BOUNDING = 0.1


@jjit
def _get_truncnorm_scale_boost(params, t_obs, t_0):
    loc_t_crit = t_0 / 2.0
    boost = _sigmoid(t_obs, loc_t_crit, LOC_T_OBS_K, params.tpck_scale_boost_early, 0.0)
    return boost


@jjit
def _get_truncnorm_scale(params, lgm):
    scale = _sigmoid(
        lgm, TPCK_LGM0, TPCK_K, params.tpck_scale_mh_lo, params.tpck_scale_mh_hi
    )
    return scale


@jjit
def mc_tpeak_cens(params, ran_key, lgmparr, tobsarr, t_0):
    utp_loc, utp_scale = _get_singlepdf_params(params, lgmparr, tobsarr, t_0)
    kern_args = tpk.UTP_Params(utp_loc, utp_scale)
    utp = tpk.mc_tp_pdf_satpop(ran_key, kern_args)
    t_peak = utp * t_0
    return t_peak


@jjit
def mc_tpeak_singlecen(params, ran_key, lgmparr, tobsarr, t_0):
    utp_loc, utp_scale = _get_singlepdf_params(params, lgmparr, tobsarr, t_0)
    kern_args = tpk.UTP_Params(utp_loc, utp_scale)
    utp = tpk.mc_tp_pdf_singlesat(ran_key, kern_args)
    t_peak = utp * t_0
    return t_peak


@jjit
def _get_singlepdf_params(tpc_params, lgm_obs, t_obs, t_0):
    scale_t0 = _get_truncnorm_scale(tpc_params, lgm_obs)
    scale = scale_t0 + _get_truncnorm_scale_boost(tpc_params, t_obs, t_0)
    loc = 0.0 * scale + DEFAULT_TRUNCNORM_LOC
    pdf_params = loc, scale
    return pdf_params


@jjit
def _t_peak_cen_pdf_singlehalo_kern(params, lgm_obs, t_obs, t_peak, t_0):
    loc = DEFAULT_TRUNCNORM_LOC
    scale_t0 = _get_truncnorm_scale(params, lgm_obs)
    scale = scale_t0 + _get_truncnorm_scale_boost(params, t_obs, t_0)
    pdf_params = loc, scale
    pdf = _t_peak_cens_pdf_kern(pdf_params, t_peak, t_0)
    return pdf


@jjit
def _t_peak_cen_cdf_singlehalo_kern(params, lgm_obs, t_obs, t_peak, t_0):
    loc = DEFAULT_TRUNCNORM_LOC
    scale_t0 = _get_truncnorm_scale(params, lgm_obs)
    scale = scale_t0 + _get_truncnorm_scale_boost(params, t_obs, t_0)
    pdf_params = loc, scale
    cdf = _t_peak_cens_cdf_kern(pdf_params, t_peak, t_0)
    return cdf


def t_peak_cens_fitter(
    x_target, pdf_target, t_0, u_p_init=tpk.DEFAULT_UTP_U_PARAMS, nstep=200, n_warmup=1
):
    loss_data = x_target, pdf_target, t_0
    args = (loss_and_grads_kern, u_p_init, loss_data)

    u_p_best, loss_best, fit_terminates, code_used = bfgs_adam_fallback(
        *args, nstep=nstep, n_warmup=n_warmup
    )
    u_p_best = tpk.UTP_UParams(*u_p_best)
    p_best = tpk.get_bounded_utp_params(u_p_best)
    return p_best, loss_best, fit_terminates, code_used


@jjit
def _t_peak_cens_pdf_kern(params, t_peak, t_0):
    loc, scale = params
    pdf_pred = tpk._tp_pdf_kern(t_peak / t_0, (loc, scale)) / t_0
    return pdf_pred


@jjit
def _t_peak_cens_cdf_kern(params, t_peak, t_0):
    loc, scale = params
    pdf_pred = tpk._tp_cdf_kern(t_peak / t_0, (loc, scale)) / t_0
    return pdf_pred


@jjit
def _loss_kern(u_params, loss_data):
    u_params = tpk.UTP_UParams(*u_params)
    params = tpk.get_bounded_utp_params(u_params)
    t_peak_target, pdf_target, t_0 = loss_data
    pdf_pred = _t_peak_cens_pdf_kern(params, t_peak_target, t_0)
    return tpk._mse(pdf_pred, pdf_target)


loss_and_grads_kern = jjit(value_and_grad(_loss_kern))


@jjit
def _get_bounded_tp_cens_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_tp_cens_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_tp_cens_params_kern = jjit(vmap(_get_bounded_tp_cens_param, in_axes=_C))
_get_unbounded_tp_cens_params_kern = jjit(
    vmap(_get_unbounded_tp_cens_param, in_axes=_C)
)


@jjit
def get_bounded_tp_cens_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _TPCens_UPNAMES])
    params = _get_bounded_tp_cens_params_kern(
        jnp.array(u_params), jnp.array(TPCENS_PBOUNDS)
    )
    params = TPCens_Params(*params)
    return params


@jjit
def get_unbounded_tp_cens_params(params):
    params = jnp.array([getattr(params, pname) for pname in TPCens_Params._fields])
    u_params = _get_unbounded_tp_cens_params_kern(
        jnp.array(params), jnp.array(TPCENS_PBOUNDS)
    )
    u_params = TPCens_UParams(*u_params)
    return u_params


DEFAULT_TPCENS_U_PARAMS = TPCens_UParams(
    *get_unbounded_tp_cens_params(DEFAULT_TPCENS_PARAMS)
)
