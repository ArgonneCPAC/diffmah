"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from ...bfgs_wrapper import diffmah_fitter
from ...utils import _inverse_sigmoid, _sig_slope, _sigmoid

DEFAULT_LGM0POP_C1_PDICT = OrderedDict(
    lgm0pop_c1_ytp=0.004,
    lgm0pop_c1_ylo=-0.030,
    lgm0pop_c1_clip_x0=5.102,
    lgm0pop_c1_clip_ylo=0.143,
    lgm0pop_c1_clip_yhi=0.002,
)
LGM0Pop_C1_Params = namedtuple("LGM0Pop_C1_Params", DEFAULT_LGM0POP_C1_PDICT.keys())
DEFAULT_LGM0POP_C1_PARAMS = LGM0Pop_C1_Params(**DEFAULT_LGM0POP_C1_PDICT)


LGM0POP_C1_BOUNDS_DICT = OrderedDict(
    lgm0pop_c1_ytp=(0.001, 0.1),
    lgm0pop_c1_ylo=(-0.05, -0.001),
    lgm0pop_c1_clip_x0=(4.0, 11.0),
    lgm0pop_c1_clip_ylo=(0.02, 0.15),
    lgm0pop_c1_clip_yhi=(0.001, 0.05),
)
LGM0POP_C1_BOUNDS = LGM0Pop_C1_Params(**LGM0POP_C1_BOUNDS_DICT)

_C1_UPNAMES = ["u_" + key for key in LGM0Pop_C1_Params._fields]
LGM0Pop_C1_UParams = namedtuple("LGM0Pop_C1_UParams", _C1_UPNAMES)

XTP = 10.0
GLOBAL_X0 = 8.0
GLOBAL_K = 0.25
CLIP_TP_K = 1.0
K_BOUNDING = 0.1


@jjit
def _pred_c1_kern(params, t_obs, t_peak):
    c1_ytp, c1_ylo, c1_clip_x0, c1_clip_ylo, c1_clip_yhi = params
    pred_c1 = _sig_slope(t_obs, XTP, c1_ytp, GLOBAL_X0, GLOBAL_K, c1_ylo, 0.0)

    clip = _sigmoid(t_peak, c1_clip_x0, CLIP_TP_K, c1_clip_ylo, c1_clip_yhi)
    pred_c1 = jnp.clip(pred_c1, a_min=clip)
    return pred_c1


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_kern_scalar(params, loss_data):
    t_obs, t_peak, target_c1 = loss_data
    pred_c1 = _pred_c1_kern(params, t_obs, t_peak)
    return _mse(target_c1, pred_c1)


@jjit
def global_loss_kern(params, global_loss_data):
    loss = 0.0
    for loss_data in global_loss_data:
        loss = loss + _loss_kern_scalar(params, loss_data)
    return loss


global_loss_and_grads_kern = jjit(value_and_grad(global_loss_kern))


def fit_global_c1_model(global_loss_data, p_init=DEFAULT_LGM0POP_C1_PARAMS):
    _res = diffmah_fitter(global_loss_and_grads_kern, p_init, global_loss_data)
    p_best, loss_best, fit_terminates, code_used = _res
    return p_best, loss_best, fit_terminates, code_used


@jjit
def _get_bounded_c1_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_c1_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_c1_params_kern = jjit(vmap(_get_bounded_c1_param, in_axes=_C))
_get_unbounded_c1_params_kern = jjit(vmap(_get_unbounded_c1_param, in_axes=_C))


@jjit
def get_bounded_c1_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _C1_UPNAMES])
    params = _get_bounded_c1_params_kern(
        jnp.array(u_params), jnp.array(LGM0POP_C1_BOUNDS)
    )
    params = LGM0Pop_C1_Params(*params)
    return params


@jjit
def get_unbounded_c1_params(params):
    params = jnp.array([getattr(params, pname) for pname in LGM0Pop_C1_Params._fields])
    u_params = _get_unbounded_c1_params_kern(
        jnp.array(params), jnp.array(LGM0POP_C1_BOUNDS)
    )
    u_params = LGM0Pop_C1_UParams(*u_params)
    return u_params


DEFAULT_LGM0POP_C1_U_PARAMS = LGM0Pop_C1_UParams(
    *get_unbounded_c1_params(DEFAULT_LGM0POP_C1_PARAMS)
)
