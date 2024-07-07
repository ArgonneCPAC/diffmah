"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...utils import _inverse_sigmoid, _sigmoid
from . import utp_pdf_kernels as tpk

K_BOUNDING = 0.1
K_T_OBS = 0.5
EPS = 1e-3

DEFAULT_UTP_SATPOP_PDICT = OrderedDict(
    utp_loc_tcrit=6.5,
    utp_scale_tcrit=9.5,
    utp_loc_c0_ylo=-0.7,
    utp_loc_c0_yhi=-2.2,
    utp_loc_c1_ylo=0.125,
    utp_loc_c1_yhi=0.225,
    utp_scale_c0_ylo=0.3,
    utp_scale_c0_yhi=0.8,
    utp_scale_c1_ylo=-0.01,
    utp_scale_c1_yhi=-0.04,
)
UTP_SatPop_Params = namedtuple("UTP_SatPop_Params", DEFAULT_UTP_SATPOP_PDICT.keys())
DEFAULT_UTP_SATPOP_PARAMS = UTP_SatPop_Params(**DEFAULT_UTP_SATPOP_PDICT)

DEFAULT_UTP_SATPOP_BOUNDS_DICT = OrderedDict(
    utp_loc_tcrit=(5.0, 10.0),
    utp_scale_tcrit=(5.0, 13.0),
    utp_loc_c0_ylo=(-1.2, -0.4),
    utp_loc_c0_yhi=(-3.0, -1.5),
    utp_loc_c1_ylo=(0.05, 0.2),
    utp_loc_c1_yhi=(0.05, 0.4),
    utp_scale_c0_ylo=(0.1, 0.5),
    utp_scale_c0_yhi=(0.5, 0.95),
    utp_scale_c1_ylo=(-0.05, 0.0),
    utp_scale_c1_yhi=(-0.1, 0.0),
)
UTP_SATPOP_BOUNDS = UTP_SatPop_Params(**DEFAULT_UTP_SATPOP_BOUNDS_DICT)

_UTP_SATPOP_UPNAMES = ["u_" + key for key in UTP_SatPop_Params._fields]
UTP_SatPop_UParams = namedtuple("UTP_SatPop_UParams", _UTP_SATPOP_UPNAMES)


@jjit
def _get_utp_loc_kern(params, lgmp, t_obs):
    c0 = _sigmoid(
        t_obs,
        params.utp_loc_tcrit,
        K_T_OBS,
        params.utp_loc_c0_ylo,
        params.utp_loc_c0_yhi,
    )
    c1 = _sigmoid(
        t_obs,
        params.utp_loc_tcrit,
        K_T_OBS,
        params.utp_loc_c1_ylo,
        params.utp_loc_c1_yhi,
    )
    utp_loc = c0 + c1 * lgmp
    ylo, yhi = tpk.UTP_PBOUNDS.utp_loc
    utp_loc = jnp.clip(utp_loc, ylo + EPS, yhi - EPS)
    return utp_loc


@jjit
def _get_utp_scale_kern(params, lgmp, t_obs):
    c0 = _sigmoid(
        t_obs,
        params.utp_scale_tcrit,
        K_T_OBS,
        params.utp_scale_c0_ylo,
        params.utp_scale_c0_yhi,
    )
    c1 = _sigmoid(
        t_obs,
        params.utp_scale_tcrit,
        K_T_OBS,
        params.utp_scale_c1_ylo,
        params.utp_scale_c1_yhi,
    )
    utp_scale = c0 + c1 * lgmp
    ylo, yhi = tpk.UTP_PBOUNDS.utp_scale
    utp_scale = jnp.clip(utp_scale, ylo + EPS, yhi - EPS)
    return utp_scale


@jjit
def _tp_cdf_kern_singlehalo(params, x, lgmp, t_obs):
    utp_loc = _get_utp_loc_kern(params, lgmp, t_obs)
    utp_scale = _get_utp_scale_kern(params, lgmp, t_obs)
    kern_args = (utp_loc, utp_scale)
    cdf = tpk._tp_cdf_kern(x, kern_args)
    return cdf


@jjit
def _tp_pdf_kern_singlehalo(params, x, lgmp, t_obs):
    utp_loc = _get_utp_loc_kern(params, lgmp, t_obs)
    utp_scale = _get_utp_scale_kern(params, lgmp, t_obs)
    kern_args = utp_loc, utp_scale
    pdf = tpk._tp_pdf_kern(x, kern_args)
    return pdf


@jjit
def mc_utp_pdf(params, ran_key, lgmparr, tobsarr):
    utp_loc = _get_utp_loc_kern(params, lgmparr, tobsarr)
    utp_scale = _get_utp_scale_kern(params, lgmparr, tobsarr)
    kern_args = tpk.UTP_Params(utp_loc, utp_scale)
    utp = tpk.mc_tp_pdf_satpop(ran_key, kern_args)
    return utp


@jjit
def _get_bounded_utp_satpop_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_utp_satpop_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_utp_satpop_params_kern = jjit(
    vmap(_get_bounded_utp_satpop_param, in_axes=_C)
)
_get_unbounded_utp_satpop_params_kern = jjit(
    vmap(_get_unbounded_utp_satpop_param, in_axes=_C)
)


@jjit
def get_bounded_utp_satpop_params(u_params):
    u_params = jnp.array(
        [getattr(u_params, u_pname) for u_pname in _UTP_SATPOP_UPNAMES]
    )
    params = _get_bounded_utp_satpop_params_kern(
        jnp.array(u_params), jnp.array(UTP_SATPOP_BOUNDS)
    )
    utp_params = UTP_SatPop_Params(*params)
    return utp_params


@jjit
def get_unbounded_utp_satpop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_UTP_SATPOP_PARAMS._fields]
    )
    u_params = _get_unbounded_utp_satpop_params_kern(
        jnp.array(params), jnp.array(UTP_SATPOP_BOUNDS)
    )
    utp_u_params = UTP_SatPop_UParams(*u_params)
    return utp_u_params


DEFAULT_UTP_SATPOP_U_PARAMS = UTP_SatPop_UParams(
    *get_unbounded_utp_satpop_params(DEFAULT_UTP_SATPOP_PARAMS)
)
