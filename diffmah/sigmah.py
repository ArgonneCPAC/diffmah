"""Model for mass assembly of individual dark matter halos."""
from collections import OrderedDict
import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import grad
from jax import vmap as jvmap
from .utils import get_1d_arrays

DEFAULT_MAH_PARAMS = OrderedDict(mah_x0=-0.15, mah_k=4.0)


@jjit
def _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index):
    """Basic kernel controlling the relation between cumulative halo mass and time."""
    rolling_index = _sigmoid(logt, x0, k, early_index, late_index)
    log_mah = rolling_index * (logt - logtmp) + logmp
    return log_mah


@jjit
def _get_bounded_params(log_early_index, u_dy):
    """Map the unbounded parameters to 0 < early < late."""
    early_index = 10 ** log_early_index  # enforces early_index>0
    delta_index_max = early_index  # enforces late_index>0
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
def _calc_log_mah(logt, logtmp, logmp, x0, k, log10_early_index, u_dy):
    """Calculate M(t) from unbounded parameters."""
    early_index, late_index = _get_bounded_params(log10_early_index, u_dy)
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index)
    return log_mah


@jjit
def _calc_mah(time, logtmp, logmp, x0, k, log_early_index, u_dy):
    """Calculate M(t) from unbounded parameters."""
    logt = jnp.log10(time)
    return 10 ** _calc_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy)


@jjit
def _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy):
    """Calculate log_mah and implement a clip at t > tmp."""
    early_index, late_index = _get_bounded_params(log_early_index, u_dy)
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index)
    return jnp.where(log_mah > logmp, logmp, log_mah)


@jjit
def _calc_clipped_log_mah2(logt, logtmp, logmp, x0, k, early_index, late_index):
    """Same as _calc_clipped_log_mah but as a function of early, late."""
    log_mah = _rolling_plaw_log_mah(logt, logtmp, logmp, x0, k, early_index, late_index)
    return jnp.where(log_mah > logmp, logmp, log_mah)


@jjit
def _calc_clipped_mah(t, lgtmp, logmp, x0, k, log_early_index, u_dy):
    """Calculate M(t) and implement a clip at t > tmp."""
    logt = jnp.log10(t)
    return 10 ** _calc_clipped_log_mah(logt, lgtmp, logmp, x0, k, log_early_index, u_dy)


@jjit
def _calc_clipped_mah2(t, lgtmp, logmp, x0, k, early, late):
    """Same as _calc_clipped_mah but as a function of early, late."""
    logt = jnp.log10(t)
    return 10 ** _calc_clipped_log_mah2(logt, lgtmp, logmp, x0, k, early, late)


_calc_dmhdt = jjit(jvmap(grad(_calc_mah, argnums=0), in_axes=(0, *[None] * 6)))
_calc_clipped_dmhdt = jjit(
    jvmap(grad(_calc_clipped_mah, argnums=0), in_axes=(0, *[None] * 6))
)
_calc_clipped_dmhdt2 = jjit(
    jvmap(grad(_calc_clipped_mah2, argnums=0), in_axes=(0, *[None] * 6))
)


@jjit
def _get_log_mah_kern(logt, logtmp, logmp, x0, k, log_early_index, u_dy):
    """Calculate log_mah and dm/dt"""
    log_mah = _calc_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    dmhdt = _calc_dmhdt(10 ** logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    return log_mah, dmhdt / 1e9


@jjit
def _get_clipped_log_mah_kern(logt, logtmp, k, logmp, x0, log_early_index, u_dy):
    """Calculate log_mah and dm/dt"""
    log_mah = _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    dmhdt = _calc_clipped_dmhdt(10 ** logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    return log_mah, dmhdt / 1e9


@jjit
def _get_clipped_log_mah_kern2(logt, logtmp, k, logmp, x0, early, late):
    """Same as _get_clipped_log_mah_kern but as a function of early, late."""
    log_mah = _calc_clipped_log_mah2(logt, logtmp, logmp, x0, k, early, late)
    dmhdt = _calc_clipped_dmhdt2(10 ** logt, logtmp, logmp, x0, k, early, late)
    return log_mah, dmhdt / 1e9


_a = [None, *[0] * 6]
_get_clipped_log_mah_halopop = jjit(jvmap(_get_clipped_log_mah_kern2, in_axes=_a))


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


_kern = jjit(jvmap(_get_clipped_log_mah_kern2, in_axes=(None, 0, None, 0, None, 0, 0)))


def individual_halo_assembly(
    t,
    logmp,
    early_index,
    late_index,
    tmp=None,
    mah_k=DEFAULT_MAH_PARAMS["mah_k"],
    mah_x0=DEFAULT_MAH_PARAMS["mah_x0"],
):
    """Model for the mass assembly of individual dark matter halos.

    Parameters
    ----------
    t : ndarray, shape (n_times, )

    logmp : float or ndarray shape (n_halos, )

    early_index : float or ndarray shape (n_halos, )

    late_index : float or ndarray shape (n_halos, )

    tmp : float or ndarray shape (n_halos, )

    mah_k : float, optional

    mah_x0 : float, optional

    Returns
    -------
    log_mah : ndarray, shape (n_halos, n_times)

    dmhdt : ndarray, shape (n_halos, n_times)

    """
    t = np.atleast_1d(t).astype("f4")
    if tmp is None:
        tmp = t[-1]
    logmp, early, late, tmp = get_1d_arrays(logmp, early_index, late_index, tmp)
    n_halos = logmp.size

    logt = np.log10(t)
    logtmp = np.log10(tmp)
    log_mah, dmhdt = _kern(logt, logtmp, mah_k, logmp, mah_x0, early, late)
    log_mah, dmhdt = np.array(log_mah), np.array(dmhdt)

    if n_halos == 1:
        log_mah = log_mah.flatten()
        dmhdt = dmhdt.flatten()

    return log_mah, dmhdt
