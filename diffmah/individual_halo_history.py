"""Model for individual halo mass assembly based on a power-law with rolling index."""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from jax import grad


_MAH_PARS = OrderedDict(mah_x0=-0.15, mah_k=4.0, mah_lge=0.5, mah_dy=0.75)
_MAH_BOUNDS = OrderedDict(mah_x0=(-0.5, 1.0), mah_k=(1.0, 10.0), mah_lge=(0.0, 1.3))


@jjit
def _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index):
    """Basic kernel controlling the relation between cumulative halo mass and time."""
    rolling_index = _sigmoid(logt, x0, k, early_index, late_index)
    log_mah = rolling_index * (logt - logtmp) + logmp
    return log_mah


@jjit
def _rolling_plaw_log_mah_vs_time(t, logtmp, logmp, x0, k, early, late):
    logt = jnp.log10(t)
    return _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early, late)


_calc_d_log_mh_dt = jjit(
    jvmap(grad(_rolling_plaw_log_mah_vs_time, argnums=0), in_axes=(0, *[None] * 6))
)


@jjit
def _calc_halo_history(logt, logtmp, logmp, x0, k, early, late):
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early, late)
    d_log_mh_dt = _calc_d_log_mh_dt(10.0 ** logt, logtmp, logmp, x0, k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt, log_mah


@jjit
def _rolling_plaw_log_mah_lge_dy(logt, logtmp, logmp, x0, k, lge, dy):
    """Calculate M(t) from unbounded parameters."""
    early, late = _get_early_late_from_lge_dy(lge, dy)
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early, late)
    return log_mah


@jjit
def _rolling_plaw_log_mah_unbounded(logt, logtmp, logmp, u_x0, u_k, u_lge, u_dy):
    """Calculate M(t) from unbounded parameters."""
    x0, k, lge, dy = _get_bounded_params(u_x0, u_k, u_lge, u_dy)
    early, late = _get_early_late_from_lge_dy(lge, dy)
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early, late)
    return log_mah


@jjit
def _get_early_late_from_lge_dy(lge, dy):
    """Map the unbounded parameters to 0 < early < late."""
    early_index = 10 ** lge

    # enforce 0 < late_index < early_index
    delta_index_max = early_index
    delta_index = _sigmoid(dy, 0, 1, 0, delta_index_max)
    late_index = early_index - delta_index

    return early_index, late_index


def _get_bounded_params_kern(u, lo, hi):
    return _sigmoid(u, 0.0, 0.1, lo, hi)


def _get_unbounded_params_kern(u, lo, hi):
    return _inverse_sigmoid(u, 0.0, 0.1, lo, hi)


_get_bounded_params_vmap = jjit(jvmap(_get_bounded_params_kern, in_axes=(0, 0, 0)))
_get_unbounded_params_vmap = jjit(jvmap(_get_unbounded_params_kern, in_axes=(0, 0, 0)))


@jjit
def _get_bounded_params(u_x0, u_k, u_lge, u_dy):
    u_params = jnp.array((u_x0, u_k, u_lge)).astype("f4")
    lo = jnp.array([float(v[0]) for v in _MAH_BOUNDS.values()]).astype("f4")
    hi = jnp.array([float(v[1]) for v in _MAH_BOUNDS.values()])
    x0, k, lge = _get_bounded_params_vmap(u_params, lo, hi)
    dy = u_dy
    return x0, k, lge, dy


@jjit
def _get_unbounded_params(x0, k, lge, dy):
    params = jnp.array((x0, k, lge)).astype("f4")
    lo = jnp.array([float(v[0]) for v in _MAH_BOUNDS.values()]).astype("f4")
    hi = jnp.array([float(v[1]) for v in _MAH_BOUNDS.values()])
    u_x0, u_k, u_lge = _get_unbounded_params_vmap(params, lo, hi)
    u_dy = dy
    return u_x0, u_k, u_lge, u_dy


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _calc_halo_history_uparams(logt, logtmp, logmp, u_x0, u_k, u_lge, u_dy):
    x0, k, lge, dy = _get_bounded_params(u_x0, u_k, u_lge, u_dy)
    early, late = _get_early_late_from_lge_dy(lge, dy)
    return _calc_halo_history(logt, logtmp, logmp, x0, k, early, late)
