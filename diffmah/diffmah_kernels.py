"""
"""

from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import lax, nn
from jax import numpy as jnp
from jax import vmap

MAH_K = 3.5

DEFAULT_MAH_PDICT = OrderedDict(
    logm0=12.0, logtc=0.05, early_index=2.6137643, late_index=0.12692805, t_peak=14.0
)
DiffmahParams = namedtuple("DiffmahParams", list(DEFAULT_MAH_PDICT.keys()))
DEFAULT_MAH_PARAMS = DiffmahParams(*list(DEFAULT_MAH_PDICT.values()))
_MAH_PNAMES = list(DEFAULT_MAH_PDICT.keys())
_MAH_UPNAMES = ["u_" + key for key in _MAH_PNAMES]
DiffmahUParams = namedtuple("DiffmahUParams", _MAH_UPNAMES)

MAH_PBDICT = OrderedDict(
    logm0=(0.0, 17.0),
    logtc=(-1.0, 1.0),
    early_index=(0.1, 10.0),
    late_index=(0.1, 5.0),
    t_peak=(0.5, 20.0),
)
MAH_PBOUNDS = DiffmahParams(*list(MAH_PBDICT.values()))


@jjit
def mah_singlehalo(mah_params, tarr, lgt0):
    dmhdt, log_mah = _diffmah_kern(mah_params, tarr, lgt0)
    return dmhdt, log_mah


@jjit
def mah_halopop(mah_params, tarr, lgt0):
    dmhdt, log_mah = _diffmah_kern_vmap(mah_params, tarr, lgt0)
    return dmhdt, log_mah


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff * nn.sigmoid(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def _power_law_index_vs_logt(logt, logtc, early, late):
    rolling_index = _sigmoid(logt, logtc, MAH_K, early, late)
    return rolling_index


@jjit
def _rolling_plaw_vs_logt(logt, logt0, logm0, logtc, early, late):
    """Kernel of the rolling power-law between halo mass and time."""
    rolling_index = _power_law_index_vs_logt(logt, logtc, early, late)
    log_mah = rolling_index * (logt - logt0) + logm0
    return log_mah


@jjit
def _rolling_plaw_vs_t(t, logt0, logm0, logtc, early, late):
    """Convenience wrapper used to calculate d/dt of _rolling_plaw_vs_logt"""
    logt = jnp.log10(t)
    return _rolling_plaw_vs_logt(logt, logt0, logm0, logtc, early, late)


@jjit
def _log_mah_noq_kern(mah_params, t, logt0):
    logm0, logtc, early, late = mah_params[:4]
    log_mah_noq = _rolling_plaw_vs_t(t, logt0, logm0, logtc, early, late)
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
def _log_mah_kern(mah_params, t, logt0):
    log_mah_noq = _log_mah_noq_kern(mah_params, t, logt0)
    lgmhq = _log_mah_noq_kern(mah_params, mah_params.t_peak, logt0)
    msk = t < mah_params.t_peak
    log_mah = jnp.where(msk, log_mah_noq, lgmhq)  # clip growth at t_peak
    return log_mah


@jjit
def _mah_kern(mah_params, t, logt0):
    log_mah = _log_mah_kern(mah_params, t, logt0)
    mah = 10**log_mah
    return mah


_dmhdt_grad_kern_unscaled = jjit(grad(_mah_kern, argnums=1))


@jjit
def _dmhdt_kern(mah_params, t, logt0):
    dmhdt_noq = _dmhdt_noq_kern(mah_params, t, logt0)
    dmhdt = jnp.where(t > mah_params.t_peak, 0.0, dmhdt_noq)
    return dmhdt


@jjit
def _dmhdt_kern_scalar(mah_params, t, logt0):
    dmhdt_noq = _dmhdt_noq_kern_scalar(mah_params, t, logt0)
    dmhdt = jnp.where(t > mah_params.t_peak, 0.0, dmhdt_noq)
    return dmhdt


@jjit
def _diffmah_kern(mah_params, t, logt0):
    dmhdt = _dmhdt_kern(mah_params, t, logt0)
    log_mah = _log_mah_kern(mah_params, t, logt0)
    return dmhdt, log_mah


@jjit
def _diffmah_kern_scalar(mah_params, t, logt0):
    dmhdt = _dmhdt_kern_scalar(mah_params, t, logt0)
    log_mah = _log_mah_kern(mah_params, t, logt0)
    return dmhdt, log_mah


_P = (0, None, None)
_diffmah_kern_vmap = jjit(vmap(_diffmah_kern, in_axes=_P))

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
    t_peak = _get_bounded_diffmah_param(u_params.u_t_peak, MAH_PBOUNDS.t_peak)
    u_early, u_late = u_parr[2:4]
    early, late = _get_early_late(u_early, u_late)
    params = DiffmahParams(logm0, logtc, early, late, t_peak)
    return params


@jjit
def get_unbounded_mah_params(params):
    parr = jnp.array([getattr(params, pname) for pname in _MAH_PNAMES])
    early, late = parr[2:4]
    u_early, u_late = _get_u_early_late(early, late)
    u_logm0 = _get_unbounded_diffmah_param(params.logm0, MAH_PBOUNDS.logm0)
    u_logtc = _get_unbounded_diffmah_param(params.logtc, MAH_PBOUNDS.logtc)
    u_t_peak = _get_unbounded_diffmah_param(params.t_peak, MAH_PBOUNDS.t_peak)
    u_params = DiffmahUParams(u_logm0, u_logtc, u_early, u_late, u_t_peak)
    return u_params


DEFAULT_MAH_U_PARAMS = DiffmahUParams(*get_unbounded_mah_params(DEFAULT_MAH_PARAMS))


@jjit
def _log_mah_kern_u_params(mah_u_params, t, logt0):
    mah_params = get_bounded_mah_params(mah_u_params)
    return _log_mah_kern(mah_params, t, logt0)


@jjit
def _dmhdt_kern_u_params(mah_u_params, t, logt0):
    mah_params = get_bounded_mah_params(mah_u_params)
    return _dmhdt_kern(mah_params, t, logt0)


@jjit
def _diffmah_kern_u_params(mah_u_params, t, logt0):
    mah_params = get_bounded_mah_params(mah_u_params)
    dmhdt = _dmhdt_kern(mah_params, t, logt0)
    log_mah = _log_mah_kern(mah_params, t, logt0)
    return dmhdt, log_mah
