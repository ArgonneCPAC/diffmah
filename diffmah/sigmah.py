"""
"""
import numpy as np
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import grad
from jax import vmap as jvmap
from .utils import get_1d_arrays


BOUNDS = OrderedDict(mah_early_index=(0, 10))
MAH_PARAMS = OrderedDict(
    mah_x0=-0.15, mah_k=4.0, mah_early_index=3.0, mah_late_index=1.0
)
MEAN_MAH_PARAMS = OrderedDict(
    early_index_x0=14.2,
    early_index_k=1.5,
    early_index_ylo=3,
    early_index_yhi=5,
    late_index_x0=14.5,
    late_index_k=0.5,
    late_index_ylo=0.15,
    late_index_yhi=3.65,
)
TODAY = 13.8


@jjit
def _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, u_ymin, u_dy):
    """
    """
    early_index, late_index = _get_bounded_params(u_ymin, u_dy)
    return _rolling_plaw(logt, logtmp, logmp, x0, k, early_index, late_index)


@jjit
def _rolling_plaw(logt, logtmp, logmp, x0, k, ymin, ymax):
    p = _sigmoid(logt, x0, k, ymin, ymax)
    log_mah = p * (logt - logtmp) + logmp
    return log_mah


@jjit
def _get_bounded_params(u_early_index, u_dy):
    min_early_index, max_early_index = BOUNDS["mah_early_index"]
    early_index = _sigmoid(u_early_index, 0, 1, min_early_index, max_early_index)
    delta_index_max = early_index - min_early_index
    delta_index = _sigmoid(u_dy, 0, 1, 0, delta_index_max)
    late_index = early_index - delta_index
    return early_index, late_index


@jjit
def _get_unbounded_params(early_index, late_index):
    u_early_index = _inverse_sigmoid(early_index, 0, 1, *BOUNDS["mah_early_index"])
    min_early_index = BOUNDS["mah_early_index"][0]
    delta_index_max = early_index - min_early_index
    delta_index = early_index - late_index
    u_dy = _inverse_sigmoid(delta_index, 0, 1, 0, delta_index_max)
    return u_early_index, u_dy


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


def _calc_mah(time, logtmp, logmp, x0, k, u_ymin, u_dy):
    logt = jnp.log10(time)
    return 10 ** _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, u_ymin, u_dy)


_calc_dmhdt = jjit(jvmap(grad(_calc_mah, argnums=0), in_axes=(0, *[None] * 6)))


@jjit
def _get_sigmah_kern(logt, logtmp, k, logmp, x0, u_ymin, u_dy):
    log_mah = _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, u_ymin, u_dy)
    log_dmhdt = jnp.log10(_calc_dmhdt(10 ** logt, logtmp, logmp, x0, k, u_ymin, u_dy))
    return log_mah, log_dmhdt - 9.0


@jjit
def _get_sigmah(params, data):
    logt, logtmp, k = data
    logmp, x0, u_ymin, u_dy = params
    return _get_sigmah_kern(logt, logtmp, k, logmp, x0, u_ymin, u_dy)


@jjit
def _get_sigmah_halopop_kern(logt, logtmp, k, logmp, x0, early_index, late_index):
    u_early_index, u_dy = _get_unbounded_params(early_index, late_index)
    return _get_sigmah_kern(logt, logtmp, k, logmp, x0, u_early_index, u_dy)


@jjit
def _mean_early_index(lgmp, early_x0, early_k, early_ylo, early_yhi):
    return _sigmoid(lgmp, early_x0, early_k, early_ylo, early_yhi)


@jjit
def _mean_late_index(lgmp, late_x0, late_k, late_ylo, late_yhi):
    return _sigmoid(lgmp, late_x0, late_k, late_ylo, late_yhi)


@jjit
def _early_index_u_dy_covariance(mah_cov_det=0.1):
    cov = mah_cov_det * jnp.array(((1 / 4, 1), (1, 8))).astype("f4")
    return cov


_get_sigmah_halopop = jjit(jvmap(_get_sigmah_halopop_kern, in_axes=(None, *[0] * 6)))


def individual_halo_assembly_history(
    cosmic_time,
    logmp,
    tmp=TODAY,
    mah_x0=MAH_PARAMS["mah_x0"],
    mah_k=MAH_PARAMS["mah_k"],
    mah_early_index=None,
    mah_late_index=None,
):
    """Rolling power-law model for the mass assembly of individual halos.

    Parameters
    ----------
    cosmic_time : ndarray of shape (n_times, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    logmp : float or ndarray of shape (n_mass, )
        Base-10 log of peak halo mass in units of Msun

    tmp : float or ndarray of shape (n_mass, ), optional
        Age of the universe in Gyr at the time halo mass attains the input logmp.

    mah_x0 : float or ndarray of shape (n_mass, ), optional
        Base-10 log of the time of peak star formation.

    mah_k : float or ndarray of shape (n_mass, ), optional
        Transition speed between early- and late-time power laws indices.

    mah_early_index : float or ndarray of shape (n_mass, ), optional
        Early-time power-law index Mh ~ (t/tmp)**mah_early_index for logt << mah_x0.

    mah_late_index : float or ndarray of shape (n_mass, ), optional
        Late-time power-law index Mh ~ t**mah_late_index for logt >> mah_x0.

    Returns
    -------
    log_mah : ndarray of shape (n_mass, n_times)
        Base-10 log of halo mass at the input times.
        Halo mass is in units of Msun.
        If n_mass = 1, log_mah will be flattened to have shape (n_times, )

    log_dmhdt : ndarray of shape (n_mass, n_times)
        Base-10 log of halo mass accretion rate at the input times
        Accretion rate is in units of Msun/yr.
        If n_mass = 1, log_dmhdt will be flattened to have shape (n_times, )

    """
    lgt, logmp, lgtmp, x0, k, early, late = _process_args(
        cosmic_time, logmp, tmp, mah_x0, mah_k, mah_early_index, mah_late_index
    )
    logmah, log_dmhdt = _get_sigmah_halopop(lgt, lgtmp, k, logmp, x0, early, late)

    n_mass = logmah.shape[0]
    if n_mass == 1:
        return np.array(logmah).flatten(), np.array(log_dmhdt).flatten()
    else:
        return np.array(logmah), np.array(log_dmhdt)


def _process_args(t, logmp, tmp, x0, k, early, late):
    logt = np.atleast_1d(np.log10(t))
    logmp = np.atleast_1d(logmp)
    logtmp = np.log10(tmp)

    if early is None:
        p = [MEAN_MAH_PARAMS[key] for key in MEAN_MAH_PARAMS.keys() if "early" in key]
        early = _mean_early_index(logmp, *p)
    if late is None:
        p = [MEAN_MAH_PARAMS[key] for key in MEAN_MAH_PARAMS.keys() if "late" in key]
        late = _mean_late_index(logmp, *p)

    logmp, logtmp, x0, k, early, late = get_1d_arrays(logmp, logtmp, x0, k, early, late)
    return logt, logmp, logtmp, x0, k, early, late
