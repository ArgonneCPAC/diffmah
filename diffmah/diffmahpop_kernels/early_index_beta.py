"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from ..diffmah_kernels import MAH_PBOUNDS
from ..utils import _inverse_sigmoid, _sigmoid

EPS = 1e-3

EARLY_INDEX_K = 1.0
EARLY_INDEX_YHI_K = 0.5
EARLY_INDEX_YLO_K = 1.0

EARLY_INDEX_PDICT = OrderedDict(
    early_index_ylo_x0_beta=1.456,
    early_index_ylo_ylo_beta=1.037,
    early_index_ylo_yhi_beta=0.879,
    early_index_yhi_ylo_beta=2.760,
    early_index_yhi_yhi_beta=2.195,
    early_index_yhi_x0_beta=8.144,
    early_index_lgm_x0_beta=12.641,
)
EARLY_INDEX_BOUNDS_PDICT = OrderedDict(
    early_index_ylo_x0_beta=(1.0, 7.0),
    early_index_ylo_ylo_beta=(1.0, 3.5),
    early_index_ylo_yhi_beta=(0.5, 3.0),
    early_index_yhi_ylo_beta=(2.0, 4.0),
    early_index_yhi_yhi_beta=(1.5, 4.0),
    early_index_yhi_x0_beta=(4.0, 13.0),
    early_index_lgm_x0_beta=(12.0, 14.0),
)
EarlyIndex_Params = namedtuple("EarlyIndex_Params", EARLY_INDEX_PDICT.keys())
DEFAULT_EARLY_INDEX_PARAMS = EarlyIndex_Params(**EARLY_INDEX_PDICT)
EARLY_INDEX_PBOUNDS = EarlyIndex_Params(**EARLY_INDEX_BOUNDS_PDICT)

K_BOUNDING = 0.1


@jjit
def _pred_early_index_kern(params, lgm_obs, t_obs, t_peak):
    early_index_ylo = _get_early_index_ylo(
        params.early_index_ylo_x0_beta,
        params.early_index_ylo_ylo_beta,
        params.early_index_ylo_yhi_beta,
        t_peak,
    )
    early_index_yhi = _get_early_index_yhi(
        params.early_index_yhi_x0_beta,
        params.early_index_yhi_ylo_beta,
        params.early_index_yhi_yhi_beta,
        t_obs,
    )
    early_index = _sigmoid(
        lgm_obs,
        params.early_index_lgm_x0_beta,
        EARLY_INDEX_K,
        early_index_ylo,
        early_index_yhi,
    )
    ylo, yhi = MAH_PBOUNDS.early_index
    early_index = jnp.clip(early_index, ylo + EPS, yhi - EPS)
    return early_index


@jjit
def _get_early_index_ylo(
    early_index_ylo_x0, early_index_ylo_ylo, early_index_ylo_yhi, t_peak
):
    early_index_ylo = _sigmoid(
        t_peak,
        early_index_ylo_x0,
        EARLY_INDEX_YLO_K,
        early_index_ylo_ylo,
        early_index_ylo_yhi,
    )
    return early_index_ylo


@jjit
def _get_early_index_yhi(
    early_index_yhi_x0, early_index_yhi_ylo, early_index_yhi_yhi, t_obs
):
    early_index_yhi = _sigmoid(
        t_obs,
        early_index_yhi_x0,
        EARLY_INDEX_YHI_K,
        early_index_yhi_ylo,
        early_index_yhi_yhi,
    )
    return early_index_yhi


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _loss_kern(u_params, loss_data):
    u_params = EarlyIndex_UParams(*u_params)
    params = get_bounded_early_index_params(u_params)
    lgm_obs, t_obs, t_peak, early_index_target = loss_data
    early_index_pred = _pred_early_index_kern(params, lgm_obs, t_obs, t_peak)
    return _mse(early_index_pred, early_index_target)


loss_and_grads_kern = jjit(value_and_grad(_loss_kern))


@jjit
def _get_bounded_early_index_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_early_index_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_early_index_params_kern = jjit(
    vmap(_get_bounded_early_index_param, in_axes=_C)
)
_get_unbounded_early_index_params_kern = jjit(
    vmap(_get_unbounded_early_index_param, in_axes=_C)
)


@jjit
def get_bounded_early_index_params(u_params):
    u_params = jnp.array(
        [getattr(u_params, u_pname) for u_pname in _EARLY_INDEX_UPNAMES]
    )
    params = _get_bounded_early_index_params_kern(
        jnp.array(u_params), jnp.array(EARLY_INDEX_PBOUNDS)
    )
    params = EarlyIndex_Params(*params)
    return params


@jjit
def get_unbounded_early_index_params(params):
    params = jnp.array([getattr(params, pname) for pname in EarlyIndex_Params._fields])
    u_params = _get_unbounded_early_index_params_kern(
        jnp.array(params), jnp.array(EARLY_INDEX_PBOUNDS)
    )
    u_params = EarlyIndex_UParams(*u_params)
    return u_params


_EARLY_INDEX_UPNAMES = ["u_" + key for key in EarlyIndex_Params._fields]
EarlyIndex_UParams = namedtuple("EarlyIndex_UParams", _EARLY_INDEX_UPNAMES)
DEFAULT_EARLY_INDEX_U_PARAMS = get_unbounded_early_index_params(
    DEFAULT_EARLY_INDEX_PARAMS
)
