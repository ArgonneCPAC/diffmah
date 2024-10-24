"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from ...bfgs_wrapper import bfgs_adam_fallback
from ...utils import _inverse_sigmoid, _sig_slope, _sigmoid

DEFAULT_LGM0POP_C0_PDICT = OrderedDict(
    lgm0pop_c0_ytp_late_sats=0.011,
    lgm0pop_c0_ylo_late_sats=-0.080,
    lgm0pop_c0_clip_c0_late_sats=0.521,
    lgm0pop_c0_clip_c1_late_sats=-0.043,
    lgm0pop_c0_t_obs_x0_late_sats=1.581,
)
LGM0Pop_C0_Params = namedtuple("LGM0Pop_C0_Params", DEFAULT_LGM0POP_C0_PDICT.keys())
DEFAULT_LGM0POP_C0_PARAMS = LGM0Pop_C0_Params(**DEFAULT_LGM0POP_C0_PDICT)

_C0_UPNAMES = ["u_" + key for key in LGM0Pop_C0_Params._fields]
LGM0Pop_C0_UParams = namedtuple("LGM0Pop_C0_UParams", _C0_UPNAMES)

LGM0POP_C0_BOUNDS_DICT = OrderedDict(
    lgm0pop_c0_ytp_late_sats=(0.01, 0.4),
    lgm0pop_c0_ylo_late_sats=(-0.15, -0.05),
    lgm0pop_c0_clip_c0_late_sats=(0.5, 0.9),
    lgm0pop_c0_clip_c1_late_sats=(-0.1, -0.01),
    lgm0pop_c0_t_obs_x0_late_sats=(1.5, 6.0),
)
LGM0POP_C0_BOUNDS = LGM0Pop_C0_Params(**LGM0POP_C0_BOUNDS_DICT)

XTP = 15
GLOBAL_K = 0.25
K_BOUNDING = 0.1


@jjit
def _pred_c0_kern(params, t_obs, t_peak):
    pred_c0 = _sig_slope(
        t_obs,
        XTP,
        params.lgm0pop_c0_ytp_late_sats,
        params.lgm0pop_c0_t_obs_x0_late_sats,
        GLOBAL_K,
        params.lgm0pop_c0_ylo_late_sats,
        0.0,
    )
    clip = (
        params.lgm0pop_c0_clip_c0_late_sats
        + params.lgm0pop_c0_clip_c1_late_sats * t_peak
    )
    pred_c0 = jnp.clip(pred_c0, min=clip)
    return pred_c0


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_kern_scalar(params, loss_data):
    t_obs, t_peak, target_c0 = loss_data
    pred_c0 = _pred_c0_kern(params, t_obs, t_peak)
    return _mse(target_c0, pred_c0)


@jjit
def global_loss_kern(params, global_loss_data):
    loss = 0.0
    for loss_data in global_loss_data:
        loss = loss + _loss_kern_scalar(params, loss_data)
    return loss


global_loss_and_grads_kern = jjit(value_and_grad(global_loss_kern))


def fit_global_c0_model(global_loss_data, p_init=DEFAULT_LGM0POP_C0_PARAMS):
    _res = bfgs_adam_fallback(global_loss_and_grads_kern, p_init, global_loss_data)
    p_best, loss_best, fit_terminates, code_used = _res
    return p_best, loss_best, fit_terminates, code_used


@jjit
def _get_bounded_c0_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_c0_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_c0_params_kern = jjit(vmap(_get_bounded_c0_param, in_axes=_C))
_get_unbounded_c0_params_kern = jjit(vmap(_get_unbounded_c0_param, in_axes=_C))


@jjit
def get_bounded_c0_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _C0_UPNAMES])
    params = _get_bounded_c0_params_kern(
        jnp.array(u_params), jnp.array(LGM0POP_C0_BOUNDS)
    )
    c0_params = LGM0Pop_C0_Params(*params)
    return c0_params


@jjit
def get_unbounded_c0_params(params):
    params = jnp.array([getattr(params, pname) for pname in LGM0Pop_C0_Params._fields])
    u_params = _get_unbounded_c0_params_kern(
        jnp.array(params), jnp.array(LGM0POP_C0_BOUNDS)
    )
    c0_u_params = LGM0Pop_C0_UParams(*u_params)
    return c0_u_params


DEFAULT_LGM0POP_C0_U_PARAMS = LGM0Pop_C0_UParams(
    *get_unbounded_c0_params(DEFAULT_LGM0POP_C0_PARAMS)
)
