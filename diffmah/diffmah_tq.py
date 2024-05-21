"""
"""

from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import lax, nn
from jax import numpy as jnp
from jax import vmap

K_MAHQ = 10.0
MAH_K = 3.5


DEFAULT_MAH_PDICT = OrderedDict(
    logm0=12.0, logtc=0.05, early_index=2.6137643, late_index=0.12692805
)
DiffmahParams = namedtuple("DiffmahParams", list(DEFAULT_MAH_PDICT.keys()))
DEFAULT_MAH_PARAMS = DiffmahParams(*list(DEFAULT_MAH_PDICT.values()))
_MAH_PNAMES = list(DEFAULT_MAH_PDICT.keys())
_MAH_UPNAMES = ["u_" + key for key in _MAH_PNAMES]
DiffmahUParams = namedtuple("DiffmahUParams", _MAH_UPNAMES)

MAH_PBDICT = OrderedDict(
    logm0=(0.0, 17.0), logtc=(-1.0, 1.0), early_index=(0.1, 10.0), late_index=(0.1, 5.0)
)
MAH_PBOUNDS = DiffmahParams(*list(MAH_PBDICT.values()))

MAH_K = 3.5


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff * nn.sigmoid(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def _power_law_index_vs_logt(logt, logtc, k, early, late):
    rolling_index = _sigmoid(logt, logtc, k, early, late)
    return rolling_index


@jjit
def _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late):
    """Kernel of the rolling power-law between halo mass and time."""
    rolling_index = _power_law_index_vs_logt(logt, logtc, k, early, late)
    log_mah = rolling_index * (logt - logt0) + logmp
    return log_mah


@jjit
def _rolling_plaw_vs_t(t, logt0, logmp, logtc, k, early, late):
    """Convenience wrapper used to calculate d/dt of _rolling_plaw_vs_logt"""
    logt = jnp.log10(t)
    return _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)


@jjit
def _log_mah_noq_kern(mah_params, t, logt0):
    logmp, logtc, early, late = mah_params
    log_mah_noq = _rolling_plaw_vs_t(t, logt0, logmp, logtc, MAH_K, early, late)
    return log_mah_noq


@jjit
def _mah_noq_kern(mah_params, t, logt0):
    log_mah_noq = _log_mah_noq_kern(mah_params, t, logt0)
    mah_noq = 10**log_mah_noq
    return mah_noq


_dmhdt_noq_grad_kern_scalar = jjit(grad(_mah_noq_kern, argnums=1))
_dmhdt_noq_grad_kern = jjit(vmap(_dmhdt_noq_grad_kern_scalar, in_axes=(None, 0, None)))


@jjit
def _dmhdt_noq_kern_scalar(mah_params, t, logt0):
    dmhdt = _dmhdt_noq_grad_kern_scalar(mah_params, t, logt0) / 1e9
    return dmhdt


_dmhdt_noq_kern = jjit(vmap(_dmhdt_noq_kern_scalar, in_axes=(None, 0, None)))


@jjit
def _log_mah_kern(mah_params, t, t_q, logt0):
    log_mah_noq = _log_mah_noq_kern(mah_params, t, logt0)
    lgmhq = _log_mah_noq_kern(mah_params, t_q, logt0)
    log_mah = jnp.where(t < t_q, log_mah_noq, lgmhq)  # clip growth at t_q
    return log_mah


@jjit
def _mah_kern(mah_params, t, t_q, logt0):
    log_mah = _log_mah_kern(mah_params, t, t_q, logt0)
    mah = 10**log_mah
    return mah


_dmhdt_grad_kern_unscaled = jjit(grad(_mah_kern, argnums=1))


@jjit
def _dmhdt_kern(mah_params, t, t_q, logt0):
    dmhdt_noq = _dmhdt_noq_kern(mah_params, t, logt0)
    dmhdt = jnp.where(t > t_q, 0.0, dmhdt_noq)
    return dmhdt


@jjit
def _diffmah_kern(mah_params, t, t_q, logt0):
    dmhdt = _dmhdt_kern(mah_params, t, t_q, logt0)
    log_mah = _log_mah_kern(mah_params, t, t_q, logt0)
    return dmhdt, log_mah


##############################
# Unbounded parameter behavior

BOUNDING_K = 0.1


@jjit
def _get_bounded_diffmah_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, BOUNDING_K, lo, hi)


@jjit
def _get_unbounded_diffmah_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, BOUNDING_K, lo, hi)


@jjit
def _get_early_late(u_early, u_late):
    late = _get_bounded_diffmah_param(u_late, MAH_PBOUNDS.late_index)
    early = _sigmoid(u_early, 0.0, BOUNDING_K, late, MAH_PBOUNDS.early_index[1])
    return early, late


@jjit
def _get_u_early_late(early, late):
    u_late = _get_unbounded_diffmah_param(late, MAH_PBOUNDS.late_index)
    u_early = _inverse_sigmoid(early, 0.0, BOUNDING_K, late, MAH_PBOUNDS.early_index[1])
    return u_early, u_late


@jjit
def get_bounded_mah_params(u_params):
    u_parr = jnp.array([getattr(u_params, u_pname) for u_pname in _MAH_UPNAMES])
    logm0 = _get_bounded_diffmah_param(u_params.u_logm0, MAH_PBOUNDS.logm0)
    logtc = _get_bounded_diffmah_param(u_params.u_logtc, MAH_PBOUNDS.logtc)
    u_early, u_late = u_parr[2:]
    early, late = _get_early_late(u_early, u_late)
    params = DiffmahParams(logm0, logtc, early, late)
    return params


@jjit
def get_unbounded_mah_params(params):
    parr = jnp.array([getattr(params, pname) for pname in _MAH_PNAMES])
    early, late = parr[2:]
    u_early, u_late = _get_u_early_late(early, late)
    u_logm0 = _get_unbounded_diffmah_param(params.logm0, MAH_PBOUNDS.logm0)
    u_logtc = _get_unbounded_diffmah_param(params.logtc, MAH_PBOUNDS.logtc)
    u_params = DiffmahUParams(u_logm0, u_logtc, u_early, u_late)
    return u_params


DEFAULT_MAH_U_PARAMS = DiffmahUParams(*get_unbounded_mah_params(DEFAULT_MAH_PARAMS))


@jjit
def _log_mah_kern_u_params(mah_u_params, t, t_q, logt0):
    mah_params = get_bounded_mah_params(mah_u_params)
    return _log_mah_kern(mah_params, t, t_q, logt0)


@jjit
def _dmhdt_kern_u_params(mah_u_params, t, t_q, logt0):
    mah_params = get_bounded_mah_params(mah_u_params)
    return _dmhdt_kern(mah_params, t, t_q, logt0)


@jjit
def _diffmah_kern_u_params(mah_u_params, t, t_q, logt0):
    mah_params = get_bounded_mah_params(mah_u_params)
    dmhdt = _dmhdt_kern(mah_params, t, t_q, logt0)
    log_mah = _log_mah_kern(mah_params, t, t_q, logt0)
    return dmhdt, log_mah
