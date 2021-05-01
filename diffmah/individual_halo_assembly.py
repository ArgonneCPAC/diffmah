"""Model for individual halo mass assembly based on a power-law with rolling index."""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap, grad

DEFAULT_MAH_PARAMS = OrderedDict(mah_logtc=0.05, mah_k=3.5, mah_ue=2.4, mah_ul=-2.0)


@jjit
def _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late):
    """Kernel of the rolling power-law between halo mass and time."""
    rolling_index = _power_law_index_vs_logt(logt, logtc, k, early, late)
    log_mah = rolling_index * (logt - logt0) + logmp
    return log_mah


@jjit
def _u_rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, ue, ul):
    """Kernel of the rolling power-law between halo mass and time."""
    early, late = _get_early_late(ue, ul)
    return _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)


@jjit
def _rolling_plaw_vs_t(t, logt0, logmp, logtc, k, early, late):
    """Convenience wrapper used to calculate d/dt of _rolling_plaw_vs_logt"""
    logt = jnp.log10(t)
    return _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)


_d_log_mh_dt = jjit(vmap(grad(_rolling_plaw_vs_t, argnums=0), in_axes=(0, *[None] * 6)))


@jjit
def _calc_halo_history(logt, logt0, logmp, logtc, k, early, late):
    log_mah = _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)
    d_log_mh_dt = _d_log_mh_dt(10.0 ** logt, logt0, logmp, logtc, k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt, log_mah


@jjit
def _softplus(x):
    return jnp.log(1 + jnp.exp(x))


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _get_early_late(ue, ul):
    late = _softplus(ul)
    early = late + _softplus(ue)
    return early, late


@jjit
def _get_ue_ul(early, late):
    ul = _inverse_softplus(late)
    ue = _inverse_softplus(early - late)
    return ue, ul


@jjit
def _power_law_index_vs_logt(logt, logtc, k, early, late):
    rolling_index = _sigmoid(logt, logtc, k, early, late)
    return rolling_index


@jjit
def _inverse_softplus(s):
    return jnp.log(jnp.exp(s) - 1.0)
