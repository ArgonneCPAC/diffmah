"""Model for P_cen(t_peak = t_0 | m_obs, t_obs)
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

K_BOUNDING = 0.1
T0_C0 = 10.0
LGM0_FTPT0 = 12.0
EPS = 1e-4

DEFAULT_FTPT0_PDICT = OrderedDict(ftpt0_c0_c0=0.081, ftpt0_c0_c1=0.041, ftpt0_c1=0.044)

FTPT0_BOUNDS = (0.01, 0.99)
FTPT0_PBOUNDS_PDICT = OrderedDict(
    ftpt0_c0_c0=FTPT0_BOUNDS, ftpt0_c0_c1=(0.01, 0.1), ftpt0_c1=(0.01, 0.25)
)


FTPT0_Params = namedtuple("FTPT0_Params", DEFAULT_FTPT0_PDICT.keys())
_FTPT0_UPNAMES = ["u_" + key for key in FTPT0_Params._fields]
FTPT0_UParams = namedtuple("FTPT0_UParams", _FTPT0_UPNAMES)


DEFAULT_FTPT0_PARAMS = FTPT0_Params(**DEFAULT_FTPT0_PDICT)
FTPT0_PBOUNDS = FTPT0_Params(**FTPT0_PBOUNDS_PDICT)


@jjit
def _ftpt0_kernel(params, lgm_obs, t_obs):
    c0 = params.ftpt0_c0_c0 + params.ftpt0_c0_c1 * (t_obs - T0_C0)
    c0 = jnp.clip(c0, *FTPT0_BOUNDS)
    c1 = params.ftpt0_c1
    ftpt0 = c0 + c1 * (lgm_obs - LGM0_FTPT0)
    ftpt0 = jnp.clip(ftpt0, *FTPT0_BOUNDS)
    return ftpt0


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def _get_bounded_ftpt0_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_ftpt0_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_ftpt0_params_kern = jjit(vmap(_get_bounded_ftpt0_param, in_axes=_C))
_get_unbounded_ftpt0_params_kern = jjit(vmap(_get_unbounded_ftpt0_param, in_axes=_C))


@jjit
def get_bounded_ftpt0_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _FTPT0_UPNAMES])
    params = _get_bounded_ftpt0_params_kern(
        jnp.array(u_params), jnp.array(FTPT0_PBOUNDS)
    )
    ftpt0_params = FTPT0_Params(*params)
    return ftpt0_params


@jjit
def get_unbounded_ftpt0_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_FTPT0_PARAMS._fields]
    )
    u_params = _get_unbounded_ftpt0_params_kern(
        jnp.array(params), jnp.array(FTPT0_PBOUNDS)
    )
    ftpt0_u_params = FTPT0_UParams(*u_params)
    return ftpt0_u_params


DEFAULT_FTPT0_U_PARAMS = FTPT0_UParams(
    *get_unbounded_ftpt0_params(DEFAULT_FTPT0_PARAMS)
)
