"""Module implementing the average_halo_assembly_history function,
which integrates a Gaussian over the dmhdt_late_index parameter that varies
the late-time power-law index, alpha, where dmh/dt ~ t**alpha
"""
import numpy as np
from .mean_um_relation_model import _prob_tmp_early_forming_cens, _frac_early_mpeak
from .mean_um_relation_model import PARAMS as UM_PARAMS
from .halo_assembly import _individual_halo_assembly_jax_kern, _get_dt_array
from .halo_assembly import DEFAULT_MAH_PARAMS, TODAY
from jax import vmap as jvmap
from jax import numpy as jnp
from jax import jit as jjit
from jax.scipy.stats import multivariate_normal as jnorm
from collections import OrderedDict


MEAN_DMHDT_EARLY_INDX_DICT = OrderedDict(
    mean_dmhdt_early_x0=13.5,
    mean_dmhdt_early_k=0.6,
    mean_dmhdt_early_ylo=0.0,
    mean_dmhdt_early_yhi=2.4,
    mean_dmhdt_early_log_scatter=-0.75,
)

MEAN_DMHDT_LATE_INDX_DICT = OrderedDict(
    mean_dmhdt_late_x0=14.25,
    mean_dmhdt_late_k=0.7,
    mean_dmhdt_late_ylo=-0.15,
    mean_dmhdt_late_yhi=-0.87,
    mean_dmhdt_late_log_scatter=-0.3,
)

MEAN_DMHDT_X0_DICT = OrderedDict(
    mean_dmhdt_x0_x0=13.91,
    mean_dmhdt_x0_k=0.48,
    mean_dmhdt_x0_ylo=-0.46,
    mean_dmhdt_x0_yhi=1.00,
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


def _clipped_halo_assembly(
    logt, dtarr, logmp, dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index, indx_tmp
):
    mah, dmhdt = _halo_assembly(
        logt,
        dtarr,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )
    mpeak = 10.0 ** logmp
    msk = mah > mpeak
    mah = jnp.where(msk, mpeak, mah)
    dmhdt = jnp.where(msk, 0.0, dmhdt)
    return mah, dmhdt


_a = (None, None, 0, 0, None, None, None, None)
_f0 = jvmap(_clipped_halo_assembly, in_axes=_a)
_a = (None, None, None, None, None, None, 0, None)
_f1 = jjit(jvmap(_f0, in_axes=_a))
_a = (None, None, None, None, None, 0, None, None)
_generate_halo_history_bundle_jax = jjit(jvmap(_f1, in_axes=_a))

_a = (None, None, None, None, None, None, None, 0)
_generate_halo_history_bundle_tmparr_jax = jjit(
    jvmap(_generate_halo_history_bundle_jax, in_axes=_a)
)


def generate_halo_history_bundle(
    t,
    logmp_grid,
    dmhdt_early_index_grid,
    dmhdt_late_index_grid,
    dmhdt_x0=DEFAULT_MAH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
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

    dmhdt_early_index_grid : ndarray, shape (n_early, )
        Values of the dmhdt_early_index halo MAH parameter

    dmhdt_late_index_grid : ndarray, shape (n_late, )
        Values of the dmhdt_late_index halo MAH parameter

    dmhdt_x0 : float, optional
        Value of the dmhdt_x0 halo MAH parameter

    dmhdt_k : float, optional
        Value of the dmhdt_k halo MAH parameter

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
        dmhdt_early_index_grid,
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
    msk = dmhdt_late_scatter < 0.01
    dmhdt_late_scatter = jnp.where(msk, 0.01, dmhdt_late_scatter)
    return jnorm.pdf(late_indx, mean_late_indx, dmhdt_late_scatter)


@jjit
def _mean_dmhdt_late_index(
    lgm, dmhdt_late_x0, dmhdt_late_k, dmhdt_late_ylo, dmhdt_late_yhi
):
    return _sigmoid(lgm, dmhdt_late_x0, dmhdt_late_k, dmhdt_late_ylo, dmhdt_late_yhi)


@jjit
def _mean_dmhdt_x0(lgm, dmhdt_x0_x0, dmhdt_x0_k, dmhdt_x0_ylo, dmhdt_x0_yhi):
    return _sigmoid(lgm, dmhdt_x0_x0, dmhdt_x0_k, dmhdt_x0_ylo, dmhdt_x0_yhi)


@jjit
def _dmhdt_early_index_pdf(
    logmp,
    early_indx,
    dmhdt_early_x0,
    dmhdt_early_k,
    dmhdt_early_ylo,
    dmhdt_early_yhi,
    dmhdt_early_scatter,
):
    mean_early_indx = _mean_dmhdt_early_index(
        logmp, dmhdt_early_x0, dmhdt_early_k, dmhdt_early_ylo, dmhdt_early_yhi
    )
    msk = dmhdt_early_scatter < 0.01
    dmhdt_early_scatter = jnp.where(msk, 0.01, dmhdt_early_scatter)
    return jnorm.pdf(early_indx, mean_early_indx, dmhdt_early_scatter)


@jjit
def _mean_dmhdt_early_index(
    lgm, dmhdt_early_x0, dmhdt_early_k, dmhdt_early_ylo, dmhdt_early_yhi
):
    return _sigmoid(
        lgm, dmhdt_early_x0, dmhdt_early_k, dmhdt_early_ylo, dmhdt_early_yhi
    )


@jjit
def _dmhdt_index_pdf(
    logmp,
    early_indx,
    late_indx,
    dmhdt_early_x0,
    dmhdt_early_k,
    dmhdt_early_ylo,
    dmhdt_early_yhi,
    dmhdt_early_scatter,
    dmhdt_late_x0,
    dmhdt_late_k,
    dmhdt_late_ylo,
    dmhdt_late_yhi,
    dmhdt_late_scatter,
):
    early_index_pdf = _dmhdt_early_index_pdf(
        logmp,
        early_indx,
        dmhdt_early_x0,
        dmhdt_early_k,
        dmhdt_early_ylo,
        dmhdt_early_yhi,
        dmhdt_early_scatter,
    )
    late_index_pdf = _dmhdt_late_index_pdf(
        logmp,
        late_indx,
        dmhdt_late_x0,
        dmhdt_late_k,
        dmhdt_late_ylo,
        dmhdt_late_yhi,
        dmhdt_late_scatter,
    )

    return early_index_pdf * late_index_pdf


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


_f = jvmap(_dmhdt_late_index_pdf, in_axes=(0, None, None, None, None, None, None))
_get_late_indx_weights_jax = jjit(
    jvmap(_f, in_axes=(None, 0, None, None, None, None, None))
)

_f = jvmap(_dmhdt_early_index_pdf, in_axes=(0, None, None, None, None, None, None))
_get_early_indx_weights_jax = jjit(
    jvmap(_f, in_axes=(None, 0, None, None, None, None, None))
)

_f = jvmap(_dmhdt_index_pdf, in_axes=(0, *[None] * 12))
_g = jvmap(_f, in_axes=(None, None, 0, *[None] * 10))
_get_indx_weights_jax = jvmap(_g, in_axes=(None, 0, *[None] * 11))


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
    return late_indx_weights
    _norm = late_indx_weights.sum(axis=0)
    n_mass = logmp_grid.size
    late_indx_weights = late_indx_weights / _norm.reshape((1, n_mass))
    return late_indx_weights


@jjit
def _get_dmhdt_late_index_norm(
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
    _norm = late_indx_weights.sum(axis=0)
    return _norm


@jjit
def _get_dmhdt_early_index_norm(
    logmp_grid,
    early_indx_grid,
    mean_dmhdt_early_x0,
    mean_dmhdt_early_k,
    mean_dmhdt_early_ylo,
    mean_dmhdt_early_yhi,
    mean_dmhdt_early_scatter,
):
    p = (
        mean_dmhdt_early_x0,
        mean_dmhdt_early_k,
        mean_dmhdt_early_ylo,
        mean_dmhdt_early_yhi,
        mean_dmhdt_early_scatter,
    )
    early_indx_weights = _get_early_indx_weights_jax(logmp_grid, early_indx_grid, *p)
    _norm = early_indx_weights.sum(axis=0)
    return _norm


@jjit
def _get_indx_weight_bundle(
    logmp_grid,
    early_indx_grid,
    late_indx_grid,
    dmhdt_early_x0,
    dmhdt_early_k,
    dmhdt_early_ylo,
    dmhdt_early_yhi,
    dmhdt_early_scatter,
    dmhdt_late_x0,
    dmhdt_late_k,
    dmhdt_late_ylo,
    dmhdt_late_yhi,
    dmhdt_late_scatter,
):
    weights = _get_indx_weights_jax(
        logmp_grid,
        early_indx_grid,
        late_indx_grid,
        dmhdt_early_x0,
        dmhdt_early_k,
        dmhdt_early_ylo,
        dmhdt_early_yhi,
        dmhdt_early_scatter,
        dmhdt_late_x0,
        dmhdt_late_k,
        dmhdt_late_ylo,
        dmhdt_late_yhi,
        dmhdt_late_scatter,
    )

    early_indx_norm = _get_dmhdt_early_index_norm(
        logmp_grid,
        early_indx_grid,
        dmhdt_early_x0,
        dmhdt_early_k,
        dmhdt_early_ylo,
        dmhdt_early_yhi,
        dmhdt_early_scatter,
    )
    late_indx_norm = _get_dmhdt_late_index_norm(
        logmp_grid,
        late_indx_grid,
        dmhdt_late_x0,
        dmhdt_late_k,
        dmhdt_late_ylo,
        dmhdt_late_yhi,
        dmhdt_late_scatter,
    )
    return weights / early_indx_norm / late_indx_norm


def get_dmhdt_late_index_weight_bundle(
    logmp_grid,
    late_indx_grid,
    mean_dmhdt_late_x0=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_x0"],
    mean_dmhdt_late_k=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_k"],
    mean_dmhdt_late_ylo=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_ylo"],
    mean_dmhdt_late_yhi=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_yhi"],
    mean_dmhdt_late_scatter=10
    ** MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_log_scatter"],
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


def average_halo_assembly_history(
    cosmic_time,
    logmp_grid,
    dmhdt_early_index_grid,
    dmhdt_late_index_grid,
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
    tmp=None,
    mean_dmhdt_x0_x0=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_x0"],
    mean_dmhdt_x0_k=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_k"],
    mean_dmhdt_x0_ylo=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_ylo"],
    mean_dmhdt_x0_yhi=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_yhi"],
    mean_dmhdt_early_x0=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_x0"],
    mean_dmhdt_early_k=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_k"],
    mean_dmhdt_early_ylo=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_ylo"],
    mean_dmhdt_early_yhi=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_yhi"],
    mean_dmhdt_early_scatter=10
    ** MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_log_scatter"],
    mean_dmhdt_late_x0=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_x0"],
    mean_dmhdt_late_k=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_k"],
    mean_dmhdt_late_ylo=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_ylo"],
    mean_dmhdt_late_yhi=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_yhi"],
    mean_dmhdt_late_scatter=10
    ** MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_log_scatter"],
):
    """
    Calculate mean halo history, averaging over Gaussian-distributed dmhdt_late_index.

    Parameters
    ----------
    cosmic_time : ndarray, shape (n_times, )
        Age of the Universe in Gyr

    logmp_grid : ndarray, shape (n_mass, )
        Base-10 log of peak halo mass in Msun

    dmhdt_early_index_grid : ndarray, shape (n_early, )
        Values of the dmhdt_early_index halo MAH parameter

    dmhdt_late_index_grid : ndarray, shape (n_late, )
        Values of the dmhdt_early_index halo MAH parameter

    dmhdt_x0 : float, optional
        Value of the dmhdt_x0 halo MAH parameter

    dmhdt_k : float, optional
        Value of the dmhdt_k halo MAH parameter

    tmp : float, optional
        Time of peak halo mass. Default value is the final time in the input t array.

    mean_dmhdt_early_x0 : float, optional
        Value of the mean_dmhdt_early_x0 parameter controlling mass-dependence of
        the average value of dmhdt_early_index

    mean_dmhdt_early_k : float, optional
        Value of the mean_dmhdt_early_k parameter controlling mass-dependence of
        the average value of dmhdt_early_index

    mean_dmhdt_early_ylo : float, optional
        Value of the mean_dmhdt_early_ylo parameter controlling mass-dependence of
        the average value of dmhdt_early_index

    mean_dmhdt_early_yhi : float, optional
        Value of the mean_dmhdt_early_yhi parameter controlling mass-dependence of
        the average value of dmhdt_early_index

    mean_dmhdt_early_scatter : float, optional
        Value of the mean_dmhdt_early_scatter parameter controlling mass-dependence of
        the average value of dmhdt_early_index

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
        dmhdt_k,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        indx_tmp,
        mean_dmhdt_x0_x0,
        mean_dmhdt_x0_k,
        mean_dmhdt_x0_ylo,
        mean_dmhdt_x0_yhi,
        mean_dmhdt_early_x0,
        mean_dmhdt_early_k,
        mean_dmhdt_early_ylo,
        mean_dmhdt_early_yhi,
        mean_dmhdt_early_scatter,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )
    return np.array(avg_log_mah), np.array(avg_log_dmhdt)


@jjit
def _avg_assembly_history(
    logt,
    dtarr,
    logmp_grid,
    dmhdt_k,
    dmhdt_early_index_grid,
    dmhdt_late_index_grid,
    indx_tmp,
    mean_dmhdt_x0_x0,
    mean_dmhdt_x0_k,
    mean_dmhdt_x0_ylo,
    mean_dmhdt_x0_yhi,
    mean_dmhdt_early_x0,
    mean_dmhdt_early_k,
    mean_dmhdt_early_ylo,
    mean_dmhdt_early_yhi,
    mean_dmhdt_early_log_scatter,
    mean_dmhdt_late_x0,
    mean_dmhdt_late_k,
    mean_dmhdt_late_ylo,
    mean_dmhdt_late_yhi,
    mean_dmhdt_late_log_scatter,
):
    mean_dmhdt_early_scatter = 10 ** mean_dmhdt_early_log_scatter
    mean_dmhdt_late_scatter = 10 ** mean_dmhdt_late_log_scatter

    dmhdt_x0_grid = _mean_dmhdt_x0(
        logmp_grid,
        mean_dmhdt_x0_x0,
        mean_dmhdt_x0_k,
        mean_dmhdt_x0_ylo,
        mean_dmhdt_x0_yhi,
    )
    bundle_generator_args = (
        logt,
        dtarr,
        logmp_grid,
        dmhdt_x0_grid,
        dmhdt_k,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        indx_tmp,
    )
    mah_bundle, dmhdt_bundle = _generate_halo_history_bundle_jax(*bundle_generator_args)

    indx_weights = _get_indx_weight_bundle(
        logmp_grid,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        mean_dmhdt_early_x0,
        mean_dmhdt_early_k,
        mean_dmhdt_early_ylo,
        mean_dmhdt_early_yhi,
        mean_dmhdt_early_scatter,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )
    n_early = dmhdt_early_index_grid.size
    n_late = dmhdt_late_index_grid.size
    n_mass = logmp_grid.size
    _W = indx_weights.reshape(n_early, n_late, n_mass, 1)
    avg_log_mah = jnp.log10(jnp.sum(jnp.sum(mah_bundle * _W, axis=0), axis=0))
    avg_log_dmhdt = jnp.log10(jnp.sum(jnp.sum(dmhdt_bundle * _W, axis=0), axis=0))
    return avg_log_mah, avg_log_dmhdt


_prob_tmp_grid = jvmap(
    _prob_tmp_early_forming_cens, in_axes=(0, None, None, None, None)
)


@jjit
def _get_tmp_weight_bundle(logmp_grid, tmp_grid, t0, tmp_k, tmp_indx_t0):
    _tmp_weights = _prob_tmp_grid(logmp_grid, tmp_grid, t0, tmp_k, tmp_indx_t0).T
    tmp_weight_norms = jnp.sum(_tmp_weights, axis=0)
    tmp_weights = _tmp_weights / tmp_weight_norms
    return tmp_weights


@jjit
def _avg_assembly_history_tmparr(
    logt,
    dtarr,
    logmp_grid,
    dmhdt_k,
    dmhdt_early_index_grid,
    dmhdt_late_index_grid,
    indx_tmp_grid,
    tmp_grid,
    today,
    mean_dmhdt_x0_x0,
    mean_dmhdt_x0_k,
    mean_dmhdt_x0_ylo,
    mean_dmhdt_x0_yhi,
    mean_dmhdt_early_x0,
    mean_dmhdt_early_k,
    mean_dmhdt_early_ylo,
    mean_dmhdt_early_yhi,
    mean_dmhdt_early_log_scatter,
    mean_dmhdt_late_x0,
    mean_dmhdt_late_k,
    mean_dmhdt_late_ylo,
    mean_dmhdt_late_yhi,
    mean_dmhdt_late_log_scatter,
    tmp_k=UM_PARAMS["tmp_k"],
    tmp_indx_t0=UM_PARAMS["tmp_indx_t0"],
):
    mean_dmhdt_early_scatter = 10 ** mean_dmhdt_early_log_scatter
    mean_dmhdt_late_scatter = 10 ** mean_dmhdt_late_log_scatter

    dmhdt_x0_grid = _mean_dmhdt_x0(
        logmp_grid,
        mean_dmhdt_x0_x0,
        mean_dmhdt_x0_k,
        mean_dmhdt_x0_ylo,
        mean_dmhdt_x0_yhi,
    )
    args = (
        logt,
        dtarr,
        logmp_grid,
        dmhdt_x0_grid,
        dmhdt_k,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        indx_tmp_grid,
    )
    mah_bundle_tmparr, dmhdt_bundle_tmparr = _generate_halo_history_bundle_tmparr_jax(
        *args
    )

    args = (
        logt,
        dtarr,
        logmp_grid,
        dmhdt_x0_grid,
        dmhdt_k,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        -1,
    )
    mah_bundle_t0, dmhdt_bundle_t0 = _generate_halo_history_bundle_jax(*args)

    indx_weights = _get_indx_weight_bundle(
        logmp_grid,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        mean_dmhdt_early_x0,
        mean_dmhdt_early_k,
        mean_dmhdt_early_ylo,
        mean_dmhdt_early_yhi,
        mean_dmhdt_early_scatter,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_scatter,
    )
    n_early = dmhdt_early_index_grid.size
    n_late = dmhdt_late_index_grid.size
    n_mass = logmp_grid.size
    n_times = logt.size

    tmp_weights = _get_tmp_weight_bundle(
        logmp_grid, tmp_grid, today, tmp_k, tmp_indx_t0
    )
    n_tmp = tmp_weights.shape[0]

    _W = indx_weights.reshape(1, n_early, n_late, n_mass, 1)
    _W2 = tmp_weights.reshape((n_tmp, 1, 1, n_mass, 1))
    w_tmparr = _W * _W2
    avg_mah_tmparr = jnp.sum(mah_bundle_tmparr * w_tmparr, axis=(0, 1, 2))
    avg_dmhdt_tmparr = jnp.sum(dmhdt_bundle_tmparr * w_tmparr, axis=(0, 1, 2))

    avg_log_mah_tmparr = jnp.log10(avg_mah_tmparr)
    log_mah_bundle_tmparr = jnp.log10(mah_bundle_tmparr)
    s = (1, 1, 1, n_mass, n_times)
    delta_log_mah_tmparr = log_mah_bundle_tmparr - avg_log_mah_tmparr.reshape(s)
    msk = jnp.isfinite(delta_log_mah_tmparr)
    delta_log_mah_tmparr = jnp.where(msk, delta_log_mah_tmparr, 0.0)
    std_mah_tmparr = jnp.sqrt(
        jnp.sum(w_tmparr * delta_log_mah_tmparr ** 2, axis=(0, 1, 2))
    )

    w_t0 = indx_weights.reshape(n_early, n_late, n_mass, 1)
    avg_mah_t0 = jnp.sum(mah_bundle_t0 * w_t0, axis=(0, 1))
    avg_dmhdt_t0 = jnp.sum(dmhdt_bundle_t0 * w_t0, axis=(0, 1))

    avg_log_mah_t0 = jnp.log10(avg_mah_t0)
    log_mah_bundle_t0 = jnp.log10(mah_bundle_t0)
    s = (1, 1, n_mass, n_times)
    delta_log_mah_t0 = log_mah_bundle_t0 - avg_log_mah_t0.reshape(s)
    msk = jnp.isfinite(delta_log_mah_t0)
    delta_log_mah_t0 = jnp.where(msk, delta_log_mah_t0, 0.0)
    std_mah_t0 = jnp.sqrt(jnp.sum(w_t0 * delta_log_mah_t0 ** 2, axis=(0, 1)))

    f_e = _frac_early_mpeak(logmp_grid).reshape((n_mass, 1))
    avg_mah = f_e * avg_mah_tmparr + (1 - f_e) * avg_mah_t0
    avg_dmhdt = f_e * avg_dmhdt_tmparr + (1 - f_e) * avg_dmhdt_t0
    avg_log_mah = jnp.log10(avg_mah)
    avg_log_dmhdt = jnp.log10(avg_dmhdt)
    std_log_mah = f_e * std_mah_tmparr + (1 - f_e) * std_mah_t0

    return avg_log_mah, avg_log_dmhdt, std_log_mah


def avg_assembly_history_tmparr(
    time_grid,
    logmp_grid,
    dmhdt_early_index_grid,
    dmhdt_late_index_grid,
    indx_tmp_grid,
    tmp_grid,
    today=TODAY,
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
    mean_dmhdt_x0_x0=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_x0"],
    mean_dmhdt_x0_k=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_k"],
    mean_dmhdt_x0_ylo=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_ylo"],
    mean_dmhdt_x0_yhi=MEAN_DMHDT_X0_DICT["mean_dmhdt_x0_yhi"],
    mean_dmhdt_early_x0=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_x0"],
    mean_dmhdt_early_k=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_k"],
    mean_dmhdt_early_ylo=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_ylo"],
    mean_dmhdt_early_yhi=MEAN_DMHDT_EARLY_INDX_DICT["mean_dmhdt_early_yhi"],
    mean_dmhdt_early_log_scatter=MEAN_DMHDT_EARLY_INDX_DICT[
        "mean_dmhdt_early_log_scatter"
    ],
    mean_dmhdt_late_x0=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_x0"],
    mean_dmhdt_late_k=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_k"],
    mean_dmhdt_late_ylo=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_ylo"],
    mean_dmhdt_late_yhi=MEAN_DMHDT_LATE_INDX_DICT["mean_dmhdt_late_yhi"],
    mean_dmhdt_late_log_scatter=MEAN_DMHDT_LATE_INDX_DICT[
        "mean_dmhdt_late_log_scatter"
    ],
    tmp_k=UM_PARAMS["tmp_k"],
    tmp_indx_t0=UM_PARAMS["tmp_indx_t0"],
):
    logt = np.log10(time_grid)
    dtarr = _get_dt_array(time_grid)

    ret = _avg_assembly_history_tmparr(
        logt,
        dtarr,
        logmp_grid,
        dmhdt_k,
        dmhdt_early_index_grid,
        dmhdt_late_index_grid,
        indx_tmp_grid,
        tmp_grid,
        today,
        mean_dmhdt_x0_x0,
        mean_dmhdt_x0_k,
        mean_dmhdt_x0_ylo,
        mean_dmhdt_x0_yhi,
        mean_dmhdt_early_x0,
        mean_dmhdt_early_k,
        mean_dmhdt_early_ylo,
        mean_dmhdt_early_yhi,
        mean_dmhdt_early_log_scatter,
        mean_dmhdt_late_x0,
        mean_dmhdt_late_k,
        mean_dmhdt_late_ylo,
        mean_dmhdt_late_yhi,
        mean_dmhdt_late_log_scatter,
        tmp_k=tmp_k,
        tmp_indx_t0=tmp_indx_t0,
    )
    avg_log_mah, avg_log_dmhdt, std_log_mah = ret
    return np.array(avg_log_mah), np.array(avg_log_dmhdt), np.array(std_log_mah)
