"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..diffmah_kernels import MAH_PBOUNDS
from ..utils import _inverse_sigmoid, _sigmoid

EPS = 1e-3
LATE_INDEX_K = 1.0

LATE_INDEX_PDICT = OrderedDict(
    late_index_x0_beta=11.567,
    late_index_ylo_beta=0.190,
    late_index_yhi_beta=0.190,
)
LATE_INDEX_BOUNDS_PDICT = OrderedDict(
    late_index_x0_beta=(11.5, 14.0),
    late_index_ylo_beta=(0.01, 0.2),
    late_index_yhi_beta=(0.01, 0.2),
)

LateIndex_Params = namedtuple("LateIndex_Params", LATE_INDEX_PDICT.keys())
DEFAULT_LATE_INDEX_PARAMS = LateIndex_Params(**LATE_INDEX_PDICT)
LATE_INDEX_PBOUNDS = LateIndex_Params(**LATE_INDEX_BOUNDS_PDICT)
K_BOUNDING = 0.1


@jjit
def _pred_late_index_kern(params, lgm_obs):
    late_index = _sigmoid(
        lgm_obs,
        params.late_index_x0_beta,
        LATE_INDEX_K,
        params.late_index_ylo_beta,
        params.late_index_yhi_beta,
    )
    ylo, yhi = MAH_PBOUNDS.late_index
    late_index = jnp.clip(late_index, ylo + EPS, yhi - EPS)
    return late_index


@jjit
def _get_bounded_late_index_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_late_index_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_late_index_params_kern = jjit(
    vmap(_get_bounded_late_index_param, in_axes=_C)
)
_get_unbounded_late_index_params_kern = jjit(
    vmap(_get_unbounded_late_index_param, in_axes=_C)
)


@jjit
def get_bounded_late_index_params(u_params):
    u_params = jnp.array(
        [getattr(u_params, u_pname) for u_pname in _LATE_INDEX_UPNAMES]
    )
    params = _get_bounded_late_index_params_kern(
        jnp.array(u_params), jnp.array(LATE_INDEX_PBOUNDS)
    )
    params = LateIndex_Params(*params)
    return params


@jjit
def get_unbounded_late_index_params(params):
    params = jnp.array([getattr(params, pname) for pname in LateIndex_Params._fields])
    u_params = _get_unbounded_late_index_params_kern(
        jnp.array(params), jnp.array(LATE_INDEX_PBOUNDS)
    )
    u_params = LateIndex_UParams(*u_params)
    return u_params


_LATE_INDEX_UPNAMES = ["u_" + key for key in LateIndex_Params._fields]
LateIndex_UParams = namedtuple("LateIndex_UParams", _LATE_INDEX_UPNAMES)
DEFAULT_LATE_INDEX_U_PARAMS = get_unbounded_late_index_params(DEFAULT_LATE_INDEX_PARAMS)
