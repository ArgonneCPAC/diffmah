"""
"""
from collections import namedtuple
import numpy as np
from jax import numpy as jnp
from jax import random as jran
from .rockstar_pdf_model import _get_mah_means_and_covs
from .individual_halo_assembly import calc_halo_history, _get_early_late
from .individual_halo_assembly import DEFAULT_MAH_PARAMS


_MCHaloPop = namedtuple(
    "MCHaloPop",
    [
        "dmhdt",
        "log_mah",
        "early_index",
        "late_index",
        "lgtc",
        "mah_type",
    ],
)


def mc_halo_population(
    cosmic_time,
    t0,
    logmh,
    mah_type=None,
    ran_key=None,
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

    logmh : float or ndarray of shape (n_halos, )
        Base-10 log of present-day halo mass of the halo population
        Units are Msun assuming h=1

    mah_type : string, optional
        Use 'early' to generate early-forming halos, 'late' for late-forming.
        Default behavior is to generate a random mixture of the two populations
        with a fraction appropriate for the input mass.

    ran_key : jax random seed, optional
        If no random key is provided, jran.PRNGKey(seed) will be chosen every time

    seed : int, optional
        Random number seed. Default is zero.

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
    n_halos = logmh.size

    cosmic_time = np.atleast_1d(cosmic_time)
    _res = _get_mah_means_and_covs(logmh, **kwargs)
    frac_late, means_early, covs_early, means_late, covs_late = _res
    n_mah_dim = means_early.shape[1]

    if ran_key is None:
        ran_key = jran.PRNGKey(seed)
    early_key, late_key, frac_key, ran_key = jran.split(ran_key, 4)

    u_p_early = jran.multivariate_normal(early_key, means_early, covs_early)
    u_p_late = jran.multivariate_normal(late_key, means_late, covs_late)
    _e = np.array(["early"])
    _l = np.array(["late"])

    if mah_type is None:
        uran = jran.uniform(frac_key, shape=(n_halos,))
        umat = jnp.repeat(uran, n_mah_dim).reshape((n_halos, n_mah_dim))
        frac_late_mat = jnp.repeat(frac_late, n_mah_dim).reshape((n_halos, n_mah_dim))
        mah_u_params = jnp.where(umat < frac_late_mat, u_p_late, u_p_early)
        mah_ue = mah_u_params[:, 0]
        mah_ul = mah_u_params[:, 1]
        mah_lgtc = mah_u_params[:, 2]
        mah_type_arr = np.where(uran < frac_late, "late", "early")
    elif mah_type == "early":
        mah_ue = u_p_early[:, 0]
        mah_ul = u_p_early[:, 1]
        mah_lgtc = u_p_early[:, 2]
        mah_type_arr = np.repeat(_e, n_halos)
    elif mah_type == "late":
        mah_ue = u_p_late[:, 0]
        mah_ul = u_p_late[:, 1]
        mah_lgtc = u_p_late[:, 2]
        mah_type_arr = np.repeat(_l, n_halos)
    else:
        msg = "`mah_type` argument = {0} but accepted values are `early` or `late`"
        raise ValueError(msg.format(mah_type))

    lgt, lgt0 = np.log10(cosmic_time), np.log10(t0)
    early, late = _get_early_late(mah_ue, mah_ul)
    _res = calc_halo_history(10 ** lgt, 10 ** lgt0, logmh, 10 ** mah_lgtc, early, late)
    dmhdt, log_mah = _res
    return _MCHaloPop(*(dmhdt, log_mah, early, late, mah_lgtc, mah_type_arr))
