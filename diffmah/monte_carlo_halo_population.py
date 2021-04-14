"""
"""
from jax import vmap
from jax import jit as jjit
from numpy.random import RandomState
import numpy as np
from .halo_population_assembly import _get_mah_means_and_covs
from .individual_halo_assembly import _calc_halo_history, DEFAULT_MAH_PARAMS

_vmap_calc_halo_history = jjit(
    vmap(_calc_halo_history, in_axes=(None, None, None, 0, None, 0, 0))
)


def mc_halo_population(
    cosmic_time,
    t0,
    logmh,
    n_halos,
    mah_type=None,
    seed=0,
    mah_k=DEFAULT_MAH_PARAMS["mah_k"],
    **kwargs
):
    """Generate Monte Carlo realization of the assembly of a population of halos.

    Parameters
    ----------
    cosmic_time : ndarray
        Array of cosmic times in units of Gyr

    t0 : float
        Present-day age of the universe in Gyr

    logmh : float
        Base-10 log of present-day halo mass of the halo population
        Units are Msun assuming h=1

    n_halos : int
        Number of halos in the population

    mah_type : string, optional
        Use 'early' to generate early-forming halos, 'late' for late-forming.
        Default behavior is to generate a random mixture of the two populations
        with a fraction appropriate for the input mass.

    seed : int, optional
        Random number seed

    **kwargs : floats
        All parameters of the MAH PDF model are accepted as keyword arguments.
        Default values are set by mah_pop_param_model.DEFAULT_MAH_PDF_PARAMS

    Returns
    -------
    dmhdt : ndarray of shape (n_halos, n_times)
        Stores halo mass accretion rate in units of Msun/yr assuming h=1

    log_mah : ndarray of shape (n_halos, n_times)
        Stores base-10 log of halo mass in units of Msun assuming h=1

    early_index : ndarray of shape (n_halos, )
        Halo MAH parameter specifying the power-law index at early times

    late_index : ndarray of shape (n_halos, )
        Halo MAH parameter specifying the power-law index at late times

    x0 : ndarray of shape (n_halos, )
        Halo MAH parameter specifying the transition time between early and late times.

    mah_type : ndarray of shape (n_halos, )
        Stores 0 for early-forming halos and 1 for late-forming

    """
    logmh = np.atleast_1d(logmh).astype("f4")
    assert logmh.size == 1, "Input halo mass must be a scalar"

    cosmic_time = np.atleast_1d(cosmic_time)

    _res = _get_mah_means_and_covs(logmh, **kwargs)
    _frac_late, _means_early, _covs_early, _means_late, _covs_late = _res
    frac_late = np.array(_frac_late[0])
    mu_early = np.array(_means_early[0])
    cov_early = np.array(_covs_early[0])
    mu_late = np.array(_means_late[0])
    cov_late = np.array(_covs_late[0])
    mah_type_arr = np.zeros(n_halos).astype("i4")

    if mah_type is None:
        u = RandomState(seed).uniform(0, 1, n_halos)
        is_late_forming = u < frac_late
        n_l = is_late_forming.sum()
        n_e = n_halos - n_l
        _e = RandomState(seed + 1).multivariate_normal(mu_early, cov_early, size=n_e)
        lge_e, lgl_e, x0_e = _e[:, 0], _e[:, 1], _e[:, 2]
        _l = RandomState(seed + 2).multivariate_normal(mu_late, cov_late, size=n_l)
        lge_l, lgl_l, x0_l = _l[:, 0], _l[:, 1], _l[:, 2]
        lge = np.concatenate((lge_e, lge_l))
        lgl = np.concatenate((lgl_e, lgl_l))
        x0 = np.concatenate((x0_e, x0_l))
        mah_type_arr[-n_l:] = 1
    elif mah_type == "early":
        data = RandomState(seed).multivariate_normal(mu_early, cov_early, size=n_halos)
        lge, lgl, x0 = data[:, 0], data[:, 1], data[:, 2]
    elif mah_type == "late":
        data = RandomState(seed).multivariate_normal(mu_late, cov_late, size=n_halos)
        lge, lgl, x0 = data[:, 0], data[:, 1], data[:, 2]
        mah_type_arr[:] = 1
    else:
        msg = "`mah_type` argument = {0} but accepted values are `early` or `late`"
        raise ValueError(msg.format(mah_type))

    lgt, lgt0 = np.log10(cosmic_time), np.log10(t0)
    _res = _vmap_calc_halo_history(lgt, lgt0, logmh[0], x0, mah_k, 10 ** lge, 10 ** lgl)
    dmhdt, log_mah = _res
    return dmhdt, log_mah, 10 ** lge, 10 ** lgl, x0, mah_type_arr
