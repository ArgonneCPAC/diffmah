"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from ..diffmah_kernels import MAH_PBOUNDS
from ..utils import _inverse_sigmoid, _sig_slope, _sigmoid

EPS = 1e-3
K_BOUNDING = 0.1
LOGTC_PDICT = OrderedDict(
    lgm_c0_tp_ytp_tobs_c0=0.395,
    lgm_c0_tp_ytp_tobs_c1=0.018,
    lgm_c0_tp_ylo=0.067,
    lgm_c0_tp_yhi=-0.071,
    lgm_c1=0.106,
)
LOGTC_BOUNDS_PDICT = OrderedDict(
    lgm_c0_tp_ytp_tobs_c0=(0.2, 0.9),
    lgm_c0_tp_ytp_tobs_c1=(0.0, 0.05),
    lgm_c0_tp_ylo=(0.02, 0.15),
    lgm_c0_tp_yhi=(-0.25, 0.0),
    lgm_c1=(0.02, 0.15),
)
Logtc_Params = namedtuple("Logtc_Params", LOGTC_PDICT.keys())
DEFAULT_LOGTC_PARAMS = Logtc_Params(**LOGTC_PDICT)
LOGTC_PBOUNDS = Logtc_Params(**LOGTC_BOUNDS_PDICT)

K_BOUNDING = 0.1
TAUC_LGMP = 12.0
C0_SS_XTP = 10.0
C0_SS_X0 = 12.0
C0_SS_K = 0.5


@jjit
def _pred_logtc_kern(params, lgm_obs, t_obs, t_peak):
    lgm_c0 = _get_c0(params, t_peak, t_obs)
    logtc = lgm_c0 + params.lgm_c1 * (lgm_obs - TAUC_LGMP)
    ylo, yhi = MAH_PBOUNDS.logtc
    logtc = jnp.clip(logtc, ylo + EPS, yhi - EPS)
    return logtc


@jjit
def _get_c0(params, t_peak, t_obs):
    ytp = params.lgm_c0_tp_ytp_tobs_c0 + params.lgm_c0_tp_ytp_tobs_c1 * t_obs
    ylo, yhi = params.lgm_c0_tp_ylo, params.lgm_c0_tp_yhi
    c0 = _sig_slope(t_peak, C0_SS_XTP, ytp, C0_SS_X0, C0_SS_K, ylo, yhi)
    return c0


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_kern(u_params, loss_data):
    u_params = Logtc_UParams(*u_params)
    params = get_bounded_logtc_params(u_params)
    lgm_obs, t_obs, t_peak, logtc_target = loss_data
    logtc_pred = _pred_logtc_kern(params, lgm_obs, t_obs, t_peak)
    return _mse(logtc_pred, logtc_target)


loss_and_grads_kern = jjit(value_and_grad(_loss_kern))


@jjit
def _get_bounded_logtc_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_logtc_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_logtc_params_kern = jjit(vmap(_get_bounded_logtc_param, in_axes=_C))
_get_unbounded_logtc_params_kern = jjit(vmap(_get_unbounded_logtc_param, in_axes=_C))


@jjit
def get_bounded_logtc_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _LOGTC_UPNAMES])
    params = _get_bounded_logtc_params_kern(
        jnp.array(u_params), jnp.array(LOGTC_PBOUNDS)
    )
    params = Logtc_Params(*params)
    return params


@jjit
def get_unbounded_logtc_params(params):
    params = jnp.array([getattr(params, pname) for pname in Logtc_Params._fields])
    u_params = _get_unbounded_logtc_params_kern(
        jnp.array(params), jnp.array(LOGTC_PBOUNDS)
    )
    u_params = Logtc_UParams(*u_params)
    return u_params


_LOGTC_UPNAMES = ["u_" + key for key in Logtc_Params._fields]
Logtc_UParams = namedtuple("Logtc_UParams", _LOGTC_UPNAMES)
DEFAULT_LOGTC_U_PARAMS = get_unbounded_logtc_params(DEFAULT_LOGTC_PARAMS)
