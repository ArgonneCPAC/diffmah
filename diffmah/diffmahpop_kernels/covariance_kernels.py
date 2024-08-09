"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import (
    _inverse_sigmoid,
    _sigmoid,
    covariance_from_correlation,
    get_cholesky_from_params,
)

COV_LGM_X0 = 13.0
COV_LGM_K = 2.0

K_BOUNDING = 0.1
DEFAULT_COV_PDICT = OrderedDict(
    std_u_logm0_ylo=0.124,
    std_u_logtc_ylo=0.242,
    std_u_early_ylo=4.975,
    std_u_late_ylo=1.752,
    std_u_logm0_yhi=0.530,
    std_u_logtc_yhi=1.037,
    std_u_early_yhi=2.998,
    std_u_late_yhi=14.612,
    rho_logtc_logm0=-0.036,
    rho_early_logm0=-0.043,
    rho_early_logtc=-0.193,
    rho_late_logm0=-0.123,
    rho_late_logtc=-0.246,
    rho_late_early=-0.167,
)

STD_BOUNDS = (0.01, 100.0)
RHO_BOUNDS = (-0.3, 0.3)
DEFAULT_COV_BOUNDS_PDICT = OrderedDict(
    std_u_logm0_ylo=STD_BOUNDS,
    std_u_logtc_ylo=STD_BOUNDS,
    std_u_early_ylo=STD_BOUNDS,
    std_u_late_ylo=STD_BOUNDS,
    std_u_logm0_yhi=STD_BOUNDS,
    std_u_logtc_yhi=STD_BOUNDS,
    std_u_early_yhi=STD_BOUNDS,
    std_u_late_yhi=STD_BOUNDS,
    rho_logtc_logm0=RHO_BOUNDS,
    rho_early_logm0=RHO_BOUNDS,
    rho_early_logtc=RHO_BOUNDS,
    rho_late_logm0=RHO_BOUNDS,
    rho_late_logtc=RHO_BOUNDS,
    rho_late_early=RHO_BOUNDS,
)


CovParams = namedtuple("CovParams", list(DEFAULT_COV_PDICT.keys()))
DEFAULT_COV_PARAMS = CovParams(**DEFAULT_COV_PDICT)
COV_PBOUNDS = CovParams(**DEFAULT_COV_BOUNDS_PDICT)


@jjit
def _get_diffmahpop_cov(params, lgm):
    diags, off_diags = _get_cov_params(params, lgm)
    ones = jnp.ones(len(diags))
    x = jnp.array((*ones, *off_diags))
    M = get_cholesky_from_params(x)
    corr_matrix = jnp.where(M == 0, M.T, M)
    cov_matrix = covariance_from_correlation(corr_matrix, jnp.array(diags))
    return cov_matrix


_get_diffmahpop_cov_vmap = jjit(vmap(_get_diffmahpop_cov, in_axes=(None, 0)))


@jjit
def _get_cov_params(params, lgm_obs):
    std_u_logm0 = _sigmoid(
        lgm_obs, COV_LGM_X0, COV_LGM_K, params.std_u_logm0_ylo, params.std_u_logm0_yhi
    )
    std_u_logtc = _sigmoid(
        lgm_obs, COV_LGM_X0, COV_LGM_K, params.std_u_logtc_ylo, params.std_u_logtc_yhi
    )
    std_u_early = _sigmoid(
        lgm_obs, COV_LGM_X0, COV_LGM_K, params.std_u_early_ylo, params.std_u_early_yhi
    )
    std_u_late = _sigmoid(
        lgm_obs, COV_LGM_X0, COV_LGM_K, params.std_u_late_ylo, params.std_u_late_yhi
    )

    diags = jnp.array((std_u_logm0, std_u_logtc, std_u_early, std_u_late))
    off_diags = jnp.array(
        [getattr(params, key) for key in params._fields if "rho_" == key[:4]]
    )

    return diags, off_diags


@jjit
def _get_bounded_cov_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_cov_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_cov_params_kern = jjit(vmap(_get_bounded_cov_param, in_axes=_C))
_get_unbounded_cov_params_kern = jjit(vmap(_get_unbounded_cov_param, in_axes=_C))


@jjit
def get_bounded_cov_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in CovUParams._fields])
    params = _get_bounded_cov_params_kern(jnp.array(u_params), jnp.array(COV_PBOUNDS))
    params = CovParams(*params)
    return params


@jjit
def get_unbounded_cov_params(params):
    params = jnp.array([getattr(params, pname) for pname in CovParams._fields])
    u_params = _get_unbounded_cov_params_kern(jnp.array(params), jnp.array(COV_PBOUNDS))
    u_params = CovUParams(*u_params)
    return u_params


CovUParams = ["u_" + key for key in CovParams._fields]
CovUParams = namedtuple("CovUParams", CovUParams)
DEFAULT_COV_U_PARAMS = get_unbounded_cov_params(DEFAULT_COV_PARAMS)
