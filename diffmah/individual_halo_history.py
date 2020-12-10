"""Model for individual halo mass assembly based on a power-law with rolling index."""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from jax import grad


DEFAULT_MAH_PARAMS = OrderedDict(mah_x0=-0.15, mah_k=4.0)


@jjit
def _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index):
    """Basic kernel controlling the relation between cumulative halo mass and time."""
    rolling_index = _sigmoid(logt, x0, k, early_index, late_index)
    log_mah = rolling_index * (logt - logtmp) + logmp
    return log_mah


@jjit
def _calc_log_mah_vs_logt(logt, logtmp, logmp, x0, k, log10_early_index, u_dy):
    """Calculate M(t) from unbounded parameters."""
    early_index, late_index = _get_bounded_params(log10_early_index, u_dy)
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index)
    return log_mah


@jjit
def _calc_log_mah_vs_time(time, logtmp, logmp, x0, k, log10_early_index, u_dy):
    logt = jnp.log10(time)
    return _calc_log_mah_vs_logt(logt, logtmp, logmp, x0, k, log10_early_index, u_dy)


@jjit
def _calc_mah(time, logtmp, logmp, x0, k, lge, u_dy):
    """Calculate M(t) from unbounded parameters."""
    logt = jnp.log10(time)
    return 10 ** _calc_log_mah_vs_logt(logt, logtmp, logmp, x0, k, lge, u_dy)


_calc_d_log_mh_dt = jjit(
    jvmap(grad(_calc_log_mah_vs_time, argnums=0), in_axes=(0, *[None] * 6))
)


@jjit
def _calc_halo_history(logt, logtmp, logmp, x0, k, lge, u_dy):
    log_mah = _calc_log_mah_vs_logt(logt, logtmp, logmp, x0, k, lge, u_dy)
    d_log_mh_dt = _calc_d_log_mh_dt(10.0 ** logt, logtmp, logmp, x0, k, lge, u_dy)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt, log_mah


@jjit
def _get_bounded_params(log_early_index, u_dy):
    """Map the unbounded parameters to 0 < early < late."""
    early_index = 10 ** log_early_index  # enforces early_index > 0
    delta_index_max = early_index  # enforces 0 < late_index < early_index
    delta_index = _sigmoid(u_dy, 0, 1, 0, delta_index_max)
    late_index = early_index - delta_index
    return early_index, late_index


@jjit
def _get_unbounded_params(early_index, late_index):
    """Map the early and late indices to the unbounded parameters.
    Input values must obey 0 < early < late."""
    log_early_index = jnp.log10(early_index)
    delta_index_max = early_index
    delta_index = early_index - late_index
    u_dy = _inverse_sigmoid(delta_index, 0, 1, 0, delta_index_max)
    return log_early_index, u_dy


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k
