"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...utils import _inverse_sigmoid, _sigmoid
from . import utp_pdf_kernels as tpk

UTP_LOC_LGM_YHI = 0.95
UTP_LOC_LGM_K = 1.5
UTP_LOC_TOBS_K = 0.4
UTP_SCALE_LGM_X0 = 12.0
UTP_SCALE_LGM_K = 1.0
UTP_SCALE_TOBS_K = 1.0
UTP_SCALE_LGM_YHI = 0.16
K_BOUNDING = 0.1

DEFAULT_TP_SATS_PDICT = OrderedDict(
    utp_loc_lgm_ylo_t0=8.779,
    utp_loc_lgm_ylo_early=0.797,
    utp_loc_lgm_ylo_late=0.160,
    utp_loc_lgm_x0=12.967,
    utp_scale_lgm_ylo_t0=6.623,
    utp_scale_lgm_ylo_early=0.105,
    utp_scale_lgm_ylo_late=0.334,
)
TP_Sats_Params = namedtuple("TP_Sats_Params", DEFAULT_TP_SATS_PDICT.keys())
DEFAULT_TP_SATS_PARAMS = TP_Sats_Params(**DEFAULT_TP_SATS_PDICT)

DEFAULT_TP_SATS_BOUNDS_DICT = OrderedDict(
    utp_loc_lgm_ylo_t0=(5.0, 10.0),
    utp_loc_lgm_ylo_early=(0.4, 0.8),
    utp_loc_lgm_ylo_late=(0.15, 0.25),
    utp_loc_lgm_x0=(11.0, 13.0),
    utp_scale_lgm_ylo_t0=(6.0, 11.0),
    utp_scale_lgm_ylo_early=(0.1, 0.5),
    utp_scale_lgm_ylo_late=(0.1, 0.6),
)
TP_SATS_BOUNDS = TP_Sats_Params(**DEFAULT_TP_SATS_BOUNDS_DICT)

_TP_SATPOP_UPNAMES = ["u_" + key for key in TP_Sats_Params._fields]
TP_Sats_UParams = namedtuple("TP_Sats_UParams", _TP_SATPOP_UPNAMES)


@jjit
def mc_utp_pdf(params, ran_key, lgmparr, tobsarr):
    utp_loc = _get_utp_loc(params, lgmparr, tobsarr)
    utp_scale = _get_utp_scale(params, lgmparr, tobsarr)
    kern_args = tpk.UTP_Params(utp_loc, utp_scale)
    utp = tpk.mc_tp_pdf_satpop(ran_key, kern_args)
    return utp


@jjit
def mc_tpeak_sats(params, ran_key, lgmparr, tobsarr):
    utp = mc_utp_pdf(params, ran_key, lgmparr, tobsarr)
    tpeak = utp * tobsarr
    return tpeak


@jjit
def mc_tpeak_singlesat(params, ran_key, lgm_obs, t_obs):
    utp_loc = _get_utp_loc(params, lgm_obs, t_obs)
    utp_scale = _get_utp_scale(params, lgm_obs, t_obs)
    kern_args = tpk.UTP_Params(utp_loc, utp_scale)
    utp = tpk.mc_tp_pdf_singlesat(ran_key, kern_args)
    tpeak = utp * t_obs
    return tpeak


@jjit
def _get_utp_loc(params, lgm_obs, t_obs):
    utp_loc_lgm_ylo = _get_utp_loc_lgm_ylo(params, t_obs)
    utp_loc = _sigmoid(
        lgm_obs, params.utp_loc_lgm_x0, UTP_LOC_LGM_K, utp_loc_lgm_ylo, UTP_LOC_LGM_YHI
    )
    return utp_loc


@jjit
def _get_utp_loc_lgm_ylo(params, t_obs):
    utp_loc_lgm_ylo = _sigmoid(
        t_obs,
        params.utp_loc_lgm_ylo_t0,
        UTP_LOC_TOBS_K,
        params.utp_loc_lgm_ylo_early,
        params.utp_loc_lgm_ylo_late,
    )
    return utp_loc_lgm_ylo


@jjit
def _get_utp_scale(params, lgm_obs, t_obs):
    utp_scale_lgm_ylo = _get_utp_scale_lgm_ylo(params, t_obs)
    utp_scale = _sigmoid(
        lgm_obs,
        UTP_SCALE_LGM_X0,
        UTP_SCALE_LGM_K,
        utp_scale_lgm_ylo,
        UTP_SCALE_LGM_YHI,
    )
    return utp_scale


@jjit
def _get_utp_scale_lgm_ylo(params, t_obs):
    utp_scale_lgm_ylo = _sigmoid(
        t_obs,
        params.utp_scale_lgm_ylo_t0,
        UTP_SCALE_TOBS_K,
        params.utp_scale_lgm_ylo_early,
        params.utp_scale_lgm_ylo_late,
    )
    return utp_scale_lgm_ylo


@jjit
def _get_bounded_tp_sat_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_tp_sat_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_tp_sat_param_kern = jjit(vmap(_get_bounded_tp_sat_param, in_axes=_C))
_get_unbounded_tp_sat_param_kern = jjit(vmap(_get_unbounded_tp_sat_param, in_axes=_C))


@jjit
def get_bounded_tp_sat_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _TP_SATPOP_UPNAMES])
    params = _get_bounded_tp_sat_param_kern(
        jnp.array(u_params), jnp.array(TP_SATS_BOUNDS)
    )
    utp_params = TP_Sats_Params(*params)
    return utp_params


@jjit
def get_unbounded_tp_sat_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_TP_SATS_PARAMS._fields]
    )
    u_params = _get_unbounded_tp_sat_param_kern(
        jnp.array(params), jnp.array(TP_SATS_BOUNDS)
    )
    utp_u_params = TP_Sats_UParams(*u_params)
    return utp_u_params


DEFAULT_TP_SATS_U_PARAMS = TP_Sats_UParams(
    *get_unbounded_tp_sat_params(DEFAULT_TP_SATS_PARAMS)
)
