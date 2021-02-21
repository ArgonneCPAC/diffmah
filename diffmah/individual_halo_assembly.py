"""Model for individual halo mass assembly based on a power-law with rolling index."""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from jax import grad


_MAH_PARS = OrderedDict(mah_x0=-0.15, mah_k=3.5, mah_lge=0.5, mah_dy=0.75)
_MAH_BOUNDS = OrderedDict(mah_x0=(-0.5, 1.0), mah_k=(1.0, 10.0), mah_lge=(0.0, 1.3))


@jjit
def _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, k, early, late):
    """Kernel of the rolling power-law between halo mass and time."""
    rolling_index = _sigmoid(logt, x0, k, early, late)
    log_mah = rolling_index * (logt - logtmp) + logmp
    return log_mah


@jjit
def _rolling_plaw_vs_t(t, logtmp, logmp, x0, k, early, late):
    """Convenience wrapper used to calculate d/dt of _rolling_plaw_vs_logt"""
    logt = jnp.log10(t)
    return _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, k, early, late)


_d_log_mh_dt = jjit(
    jvmap(grad(_rolling_plaw_vs_t, argnums=0), in_axes=(0, *[None] * 6))
)


@jjit
def _calc_halo_history(logt, logtmp, logmp, x0, k, early, late):
    log_mah = _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, k, early, late)
    d_log_mh_dt = _d_log_mh_dt(10.0 ** logt, logtmp, logmp, x0, k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt, log_mah


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))
