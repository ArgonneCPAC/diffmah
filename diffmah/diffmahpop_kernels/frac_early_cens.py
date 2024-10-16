"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

FEC_LGM_K = 4.0
FEC_T_K = 1.0
K_BOUNDING = 0.1

DEFAULT_FEC_PDICT = OrderedDict(
    fec_t_obs_x0=5.485,
    fec_lgm_x0=12.510,
    fec_lgm_ylo_early=0.546,
    fec_lgm_ylo_late=0.421,
    fec_lgm_yhi_early=0.415,
    fec_lgm_yhi_late=0.298,
)

_FBOUNDS = (1.0 / 4.0, 3.0 / 4.0)
FEC_BOUNDS_PDICT = OrderedDict(
    fec_t_obs_x0=(5.0, 13.0),
    fec_lgm_x0=(11.0, 14.0),
    fec_lgm_ylo_early=_FBOUNDS,
    fec_lgm_ylo_late=_FBOUNDS,
    fec_lgm_yhi_early=_FBOUNDS,
    fec_lgm_yhi_late=_FBOUNDS,
)
FEC_Params = namedtuple("FEC_Params", DEFAULT_FEC_PDICT.keys())
DEFAULT_FEC_PARAMS = FEC_Params(**DEFAULT_FEC_PDICT)
FEC_PBOUNDS = FEC_Params(**FEC_BOUNDS_PDICT)


@jjit
def _frac_early_cens_kern(fec_params, lgm_obs, t_obs):
    fec_lgm_ylo = _sigmoid(
        t_obs,
        fec_params.fec_t_obs_x0,
        FEC_T_K,
        fec_params.fec_lgm_ylo_early,
        fec_params.fec_lgm_ylo_late,
    )
    fec_lgm_yhi = _sigmoid(
        t_obs,
        fec_params.fec_t_obs_x0,
        FEC_T_K,
        fec_params.fec_lgm_yhi_early,
        fec_params.fec_lgm_yhi_late,
    )

    frac_early_cens = _sigmoid(
        lgm_obs, fec_params.fec_lgm_x0, FEC_LGM_K, fec_lgm_ylo, fec_lgm_yhi
    )
    return frac_early_cens


@jjit
def _get_bounded_fec_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_fec_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_fec_params_kern = jjit(vmap(_get_bounded_fec_param, in_axes=_C))
_get_unbounded_fec_params_kern = jjit(vmap(_get_unbounded_fec_param, in_axes=_C))


@jjit
def get_bounded_fec_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _FEC_UPNAMES])
    params = _get_bounded_fec_params_kern(jnp.array(u_params), jnp.array(FEC_PBOUNDS))
    params = FEC_Params(*params)
    return params


@jjit
def get_unbounded_fec_params(params):
    params = jnp.array([getattr(params, pname) for pname in FEC_Params._fields])
    u_params = _get_unbounded_fec_params_kern(jnp.array(params), jnp.array(FEC_PBOUNDS))
    u_params = FEC_UParams(*u_params)
    return u_params


_FEC_UPNAMES = ["u_" + key for key in FEC_Params._fields]
FEC_UParams = namedtuple("FEC_UParams", _FEC_UPNAMES)
DEFAULT_FEC_U_PARAMS = get_unbounded_fec_params(DEFAULT_FEC_PARAMS)
