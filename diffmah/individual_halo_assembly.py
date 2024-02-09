"""Model for individual halo mass assembly based on a power-law with rolling index."""

from jax import grad
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from .defaults import LGT0, MAH_K
from .utils import _sigmoid, get_1d_arrays


@jjit
def mah_singlehalo(mah_params, tarr, lgt0=LGT0):
    lgtarr = jnp.log10(tarr)
    dmhdt, log_mah = _calc_halo_history(
        lgtarr,
        lgt0,
        mah_params.logmp,
        mah_params.logtc,
        MAH_K,
        mah_params.early_index,
        mah_params.late_index,
    )
    return dmhdt, log_mah


@jjit
def mah_halopop(mah_params, tarr, lgt0=LGT0):
    lgtarr = jnp.log10(tarr)
    dmhdt, log_mah = _calc_halopop_history(
        lgtarr,
        lgt0,
        mah_params.logmp,
        mah_params.logtc,
        MAH_K,
        mah_params.early_index,
        mah_params.late_index,
    )
    return dmhdt, log_mah


def calc_halo_history(t, t0, logmp, tauc, early, late, k=MAH_K):
    """Calculate individual halo assembly histories.

    Parameters
    ----------
    t : ndarray of shape (n_times, )
        Age of the universe in Gyr

    t0 : float
        Present-day age of the universe in Gyr

    logmp : float
        Base-10 log of present-day peak halo mass in units of Msun assuming h=1

    tauc : float or ndarray of shape (n_halos, )
        Transition time between the fast- and slow-accretion regimes in Gyr

    early : float or ndarray of shape (n_halos, )
        Early-time power-law index in the scaling relation M(t)~t^a

    late : float or ndarray of shape (n_halos, )
        Late-time power-law index in the scaling relation M(t)~t^a

    k : float or ndarray of shape (n_halos, ), optional
        Transition speed between fast- and slow-accretion regimes.
        Default value of 3.5 is set by the DEFAULT_MAH_PARAMS dictionary

    Returns
    -------
    dmhdt : ndarray of shape (n_halos, n_times)
        Mass accretion rate in units of Msun/yr assuming h=1

    log_mah : ndarray of shape (n_halos, n_times)
        Base-10 log of cumulative peak halo mass in units of Msun assuming h=1

    """
    assert jnp.all(early > 0), "early-time index must be strictly positive"
    assert jnp.all(late > 0), "late-time index must be strictly positive"
    assert jnp.all(tauc > 0), "tauc must be strictly positive"
    lgt = jnp.log10(t)
    lgt0 = jnp.log10(t0)
    lgtc = jnp.log10(tauc)
    logmp, lgtc, k, early, late = get_1d_arrays(logmp, lgtc, k, early, late)
    dmhdt, log_mah = _vmap_halo_history(lgt, lgt0, logmp, lgtc, k, early, late)
    return dmhdt, log_mah


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


_d_log_mh_dt_scalar = jjit(grad(_rolling_plaw_vs_t, argnums=0))
_d_log_mh_dt = jjit(vmap(_d_log_mh_dt_scalar, in_axes=(0, *[None] * 6)))


@jjit
def _calc_halo_history_scalar(logt, logt0, logmp, logtc, k, early, late):
    log_mah = _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)
    d_log_mh_dt = _d_log_mh_dt_scalar(10.0**logt, logt0, logmp, logtc, k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt, log_mah


@jjit
def _calc_halo_history(logt, logt0, logmp, logtc, k, early, late):
    log_mah = _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)
    d_log_mh_dt = _d_log_mh_dt(10.0**logt, logt0, logmp, logtc, k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt, log_mah


_YO = (None, None, 0, 0, None, 0, 0)
_calc_halopop_history = jjit(vmap(_calc_halo_history, in_axes=_YO))


@jjit
def _softplus(x):
    return jnp.log(1 + lax.exp(x))


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
    return jnp.log(lax.exp(s) - 1.0)


_vmap_halo_history = jjit(vmap(_calc_halo_history, in_axes=(None, None, 0, 0, 0, 0, 0)))
