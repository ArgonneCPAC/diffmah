"""Module implementing the average_halo_assembly_history function,
which integrates a Gaussian over the dmhdt_late_index parameter that varies
the late-time power-law index, alpha, where dmh/dt ~ t**alpha
"""
import numpy as np
from diffmah.halo_assembly import _individual_halo_assembly_jax_kern, _get_dt_array
from diffmah.halo_assembly import DEFAULT_MAH_PARAMS
from jax import vmap
from jax import numpy as jnp
from jax import jit as jjit
from jax.scipy.stats import multivariate_normal as jnorm
from collections import OrderedDict


MEAN_DMHDT_LATE_INDX_DICT = OrderedDict(
    mean_dmhdt_late_x0=13,
    mean_dmhdt_late_k=1,
    mean_dmhdt_late_ylo=-1.25,
    mean_dmhdt_late_yhi=-0.5,
    mean_dmhdt_late_scatter=1.25,
)


def _halo_assembly(
    logt, dtarr, logmp, dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index, indx_tmp
):
    log_mah, log_dmhdt = _individual_halo_assembly_jax_kern(
        logt,
        dtarr,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )
    mah = 10 ** log_mah
    dmhdt = 10 ** log_dmhdt
    return mah, dmhdt


_a = (None, None, 0, None, None, None, None, None)
_f0 = vmap(_halo_assembly, in_axes=_a)

_a = (None, None, None, None, None, None, 0, None)
_generate_halo_history_bundle_jax = jjit(vmap(_f0, in_axes=_a))


def generate_halo_history_bundle(
    t,
    logmp_grid,
    dmhdt_late_index_grid,
    dmhdt_x0=DEFAULT_MAH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
    dmhdt_early_index=DEFAULT_MAH_PARAMS["dmhdt_early_index"],
    tmp=None,
):
    """Generate a grid of halo histories, one for each combination of the input grids
    of logmp and late_index, using the diffmah.halo_assembly model.

    Parameters
    ----------
    t : ndarray, shape (n_times, )
        Age of the Universe in Gyr

    logmp_grid : ndarray, shape (n_mass, )
        Base-10 log of peak halo mass in Msun

    dmhdt_late_index_grid : ndarray, shape (n_late, )
        Values of the dmhdt_early_index halo MAH parameter

    dmhdt_x0 : float, optional
        Value of the dmhdt_x0 halo MAH parameter

    dmhdt_k : float, optional
        Value of the dmhdt_k halo MAH parameter

    dmhdt_early_index : float, optional
        Value of the dmhdt_early_index halo MAH parameter

    tmp : float, optional
        Time of peak halo mass. Default value is the final time in the input t array.

    Returns
    -------
    log_mah  : ndarray, shape (n_late, n_mass, n_times)
        Base-10 log of halo mass across time for each halo in the grid

    log_dmhdt  : ndarray, shape (n_late, n_mass, n_times)
        Base-10 log of halo mass accretion rate across time for each halo in the grid

    """
    dtarr = _get_dt_array(t)
    if tmp is None:
        indx_tmp = -1
    else:
        indx_tmp = np.argmin(np.abs(t - tmp))
    logt = np.log10(t)
    args = (
        logt,
        dtarr,
        logmp_grid,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index_grid,
        indx_tmp,
    )
    log_mah, log_dmhdt = _generate_halo_history_bundle_jax(*args)
    return np.array(log_mah), np.array(log_dmhdt)


@jjit
def _dmhdt_late_index_pdf(
    logmp,
    late_indx,
    dmhdt_late_x0,
    dmhdt_late_k,
    dmhdt_late_ylo,
    dmhdt_late_yhi,
    dmhdt_late_scatter,
):
    mean_late_indx = _mean_dmhdt_late_index(
        logmp, dmhdt_late_x0, dmhdt_late_k, dmhdt_late_ylo, dmhdt_late_yhi
    )
    return jnorm.pdf(late_indx, mean_late_indx, dmhdt_late_scatter)


@jjit
def _mean_dmhdt_late_index(
    lgm, dmhdt_late_x0, dmhdt_late_k, dmhdt_late_ylo, dmhdt_late_yhi
):
    return _sigmoid(lgm, dmhdt_late_x0, dmhdt_late_k, dmhdt_late_ylo, dmhdt_late_yhi)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


_f = vmap(_dmhdt_late_index_pdf, in_axes=(None, 0, None, None, None, None, None))
_get_late_indx_weights_jax = jjit(
    vmap(_f, in_axes=(0, None, None, None, None, None, None))
)


@jjit
def _get_dmhdt_late_index_weight_bundle_jax(
    logmp_grid,
    late_indx_grid,
    mean_dmhdt_late_x0,
    mean_dmhdt_late_k,
    mean_dmhdt_late_ylo,
    mean_dmhdt_late_yhi,
    mean_dmhdt_late_scatter,
):
    p = (
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )
    late_indx_weights = _get_late_indx_weights_jax(logmp_grid, late_indx_grid, *p)
    _norm = late_indx_weights.sum(axis=1)
    n_mass = logmp_grid.size
    late_indx_weights = late_indx_weights / _norm.reshape((n_mass, 1))
    return late_indx_weights


def get_dmhdt_late_index_weight_bundle(
    logmp_grid,
    late_indx_grid,
    mean_dmhdt_late_x0=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_x0"],
    mean_dmhdt_late_k=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_k"],
    mean_dmhdt_late_ylo=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_ylo"],
    mean_dmhdt_late_yhi=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_yhi"],
    mean_dmhdt_late_scatter=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_scatter"],
):
    """
    Calculate the weights to assign to each halo when summing over the bundle to
    compute average halo histories.

    Parameters
    ----------
    logmp_grid : ndarray, shape (n_mass, )
        Base-10 log of peak halo mass in Msun

    dmhdt_late_index_grid : ndarray, shape (n_late, )
        Values of the dmhdt_early_index parameter controlling mass-dependence of
        the dmhdt_late_index parameter

    mean_dmhdt_late_x0 : float, optional
        Value of the mean_dmhdt_late_x0 parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_k : float, optional
        Value of the mean_dmhdt_late_k parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_ylo : float, optional
        Value of the mean_dmhdt_late_ylo parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_yhi : float, optional
        Value of the mean_dmhdt_late_yhi parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_scatter : float, optional
        Value of the mean_dmhdt_late_scatter parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    Returns
    -------
    weight_bundle : ndarray, shape (n_mass, n_late)
        Array of PDF values to use as weights when summing over halos in the bundle.
        The weights have been normalized so that weight_bundle.sum(axis=1) is unity.

    """
    weight_bundle = _get_dmhdt_late_index_weight_bundle_jax(
        logmp_grid,
        late_indx_grid,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )

    return np.array(weight_bundle)


@jjit
def _avg_assembly_history(
    logt,
    dtarr,
    logmp_grid,
    dmhdt_x0,
    dmhdt_k,
    dmhdt_early_index,
    dmhdt_late_index_grid,
    indx_tmp,
    mean_dmhdt_late_x0,
    mean_dmhdt_late_k,
    mean_dmhdt_late_ylo,
    mean_dmhdt_late_yhi,
    mean_dmhdt_late_scatter,
):
    bundle_generator_args = (
        logt,
        dtarr,
        logmp_grid,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index_grid,
        indx_tmp,
    )
    mah_bundle, dmhdt_bundle = _generate_halo_history_bundle_jax(*bundle_generator_args)

    late_indx_weights = _get_dmhdt_late_index_weight_bundle_jax(
        logmp_grid,
        dmhdt_late_index_grid,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )
    n_late = dmhdt_late_index_grid.size
    n_mass = logmp_grid.size
    _W = late_indx_weights.T.reshape(n_late, n_mass, 1)
    avg_log_mah = jnp.log10(jnp.sum(mah_bundle * _W, axis=0))
    avg_log_dmhdt = jnp.log10(jnp.sum(dmhdt_bundle * _W, axis=0))
    return avg_log_mah, avg_log_dmhdt


def average_halo_assembly_history(
    cosmic_time,
    logmp_grid,
    dmhdt_late_index_grid,
    dmhdt_x0=DEFAULT_MAH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
    dmhdt_early_index=DEFAULT_MAH_PARAMS["dmhdt_early_index"],
    tmp=None,
    mean_dmhdt_late_x0=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_x0"],
    mean_dmhdt_late_k=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_k"],
    mean_dmhdt_late_ylo=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_ylo"],
    mean_dmhdt_late_yhi=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_yhi"],
    mean_dmhdt_late_scatter=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_scatter"],
):
    """
    Calculate mean halo history, averaging over Gaussian-distributed dmhdt_late_index.

    Parameters
    ----------
    cosmic_time : ndarray, shape (n_times, )
        Age of the Universe in Gyr

    logmp_grid : ndarray, shape (n_mass, )
        Base-10 log of peak halo mass in Msun

    dmhdt_late_index_grid : ndarray, shape (n_late, )
        Values of the dmhdt_early_index halo MAH parameter

    dmhdt_x0 : float, optional
        Value of the dmhdt_x0 halo MAH parameter

    dmhdt_k : float, optional
        Value of the dmhdt_k halo MAH parameter

    dmhdt_early_index : float, optional
        Value of the dmhdt_early_index halo MAH parameter

    tmp : float, optional
        Time of peak halo mass. Default value is the final time in the input t array.

    mean_dmhdt_late_x0 : float, optional
        Value of the mean_dmhdt_late_x0 parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_k : float, optional
        Value of the mean_dmhdt_late_k parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_ylo : float, optional
        Value of the mean_dmhdt_late_ylo parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_yhi : float, optional
        Value of the mean_dmhdt_late_yhi parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    mean_dmhdt_late_scatter : float, optional
        Value of the mean_dmhdt_late_scatter parameter controlling mass-dependence of
        the average value of dmhdt_late_index

    Returns
    -------
    avg_log_mah  : ndarray, shape (n_mass, n_times)
        Base-10 log of halo mass across time, averaged over dmhdt_late_index

    avg_log_dmhdt  : ndarray, shape (n_late, n_mass, n_times)
        Base-10 log of halo mass accretion rate, averaged over dmhdt_late_index

    """
    logt = np.log10(cosmic_time)
    dtarr = _get_dt_array(cosmic_time)
    if tmp is None:
        indx_tmp = -1
    else:
        indx_tmp = np.argmin(np.abs(cosmic_time - tmp))

    avg_log_mah, avg_log_dmhdt = _avg_assembly_history(
        logt,
        dtarr,
        logmp_grid,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index_grid,
        indx_tmp,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )
    return np.array(avg_log_mah), np.array(avg_log_dmhdt)
