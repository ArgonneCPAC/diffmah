"""
"""
from numpy.random import RandomState
import numpy as np
from .rockstar_pdf_model import _get_mah_means_and_covs
from .individual_halo_assembly import calc_halo_history, _get_early_late
from .individual_halo_assembly import DEFAULT_MAH_PARAMS


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
        Default values are set by rockstar_pdf_model.DEFAULT_MAH_PDF_PARAMS

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

    lgtc : ndarray of shape (n_halos, )
        Halo MAH parameter specifying the transition time between early and late times.

    mah_type : ndarray of shape (n_halos, )
        Array of strings, either 'early' or 'late'

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

    _e = np.array(["early"])
    _l = np.array(["late"])

    if mah_type is None:
        u = RandomState(seed).uniform(0, 1, n_halos)
        is_late_forming = u < frac_late
        n_l = is_late_forming.sum()
        n_e = n_halos - n_l
        _e = RandomState(seed + 1).multivariate_normal(mu_early, cov_early, size=n_e)
        ue_e, ul_e, lgtc_e = _e[:, 0], _e[:, 1], _e[:, 2]
        _l = RandomState(seed + 2).multivariate_normal(mu_late, cov_late, size=n_l)
        ue_l, ul_l, lgtc_l = _l[:, 0], _l[:, 1], _l[:, 2]
        ue = np.concatenate((ue_e, ue_l))
        ul = np.concatenate((ul_e, ul_l))
        lgtc = np.concatenate((lgtc_e, lgtc_l))
        se = ["early"] * n_e
        sl = ["late"] * n_l
        se.extend(sl)
        mah_type_arr = np.array(se)
    elif mah_type == "early":
        data = RandomState(seed).multivariate_normal(mu_early, cov_early, size=n_halos)
        ue, ul, lgtc = data[:, 0], data[:, 1], data[:, 2]
        mah_type_arr = np.repeat(_e, n_halos)
    elif mah_type == "late":
        data = RandomState(seed).multivariate_normal(mu_late, cov_late, size=n_halos)
        ue, ul, lgtc = data[:, 0], data[:, 1], data[:, 2]
        mah_type_arr = np.repeat(_l, n_halos)
    else:
        msg = "`mah_type` argument = {0} but accepted values are `early` or `late`"
        raise ValueError(msg.format(mah_type))

    lgt, lgt0 = np.log10(cosmic_time), np.log10(t0)
    early, late = _get_early_late(ue, ul)
    _res = calc_halo_history(10 ** lgt, 10 ** lgt0, logmh[0], 10 ** lgtc, early, late)
    dmhdt, log_mah = _res
    return dmhdt, log_mah, early, late, lgtc, mah_type_arr
