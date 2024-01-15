"""
"""
import typing

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .defaults import MAH_K
from .individual_halo_assembly import _calc_halo_history, _get_early_late
from .rockstar_pdf_model import DEFAULT_MAH_PDF_PARAMS, _get_mah_means_and_covs

_A = (None, None, 0, 0, None, 0, 0)
_calc_halo_history_vmap = jjit(vmap(_calc_halo_history, in_axes=_A))


class MCHaloPop(typing.NamedTuple):
    dmhdt: jnp.array
    log_mah: jnp.array
    early_index: jnp.array
    late_index: jnp.array
    lgtc: jnp.array
    mah_type: jnp.array


def mc_halo_population(cosmic_time, t0, logmh, ran_key=None, seed=0, **kwargs):
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
        Integer array storing 1 for late-type and 0 for early-type halo MAHs

    """
    input_kwargs = list(kwargs.keys())
    for key in input_kwargs:
        msg = "Input parameter `{0}` not recognized".format(key)
        assert key in DEFAULT_MAH_PDF_PARAMS, msg

    logmh = np.atleast_1d(logmh).astype("f4")

    cosmic_time = np.atleast_1d(cosmic_time)
    lgt0 = np.log10(t0)

    mah_pdf_pdict = DEFAULT_MAH_PDF_PARAMS.copy()
    mah_pdf_pdict.update(kwargs)
    mah_pdf_params = np.array(list(mah_pdf_pdict.values()))

    if ran_key is None:
        ran_key = jran.PRNGKey(seed)

    halopop = _mc_halo_mahs(ran_key, cosmic_time, lgt0, logmh, mah_pdf_params)
    return halopop


@jjit
def _mc_halo_mahs(ran_key, tarr, lgt0, lgm0, mah_pdf_params):
    early_key, late_key, frac_key, ran_key = jran.split(ran_key, 4)

    _res = _get_mah_means_and_covs(lgm0, *mah_pdf_params)
    frac_late, means_early, covs_early, means_late, covs_late = _res

    mah_u_params_early = jran.multivariate_normal(early_key, means_early, covs_early)
    mah_u_params_late = jran.multivariate_normal(late_key, means_late, covs_late)

    uran = jran.uniform(frac_key, shape=(lgm0.size,))
    msk_is_late = jnp.where(uran < frac_late, 1, 0)

    mah_ue = jnp.where(msk_is_late, mah_u_params_late[:, 0], mah_u_params_early[:, 0])
    mah_ul = jnp.where(msk_is_late, mah_u_params_late[:, 1], mah_u_params_early[:, 1])
    mah_lgtc = jnp.where(msk_is_late, mah_u_params_late[:, 2], mah_u_params_early[:, 2])

    early, late = _get_early_late(mah_ue, mah_ul)

    lgtarr = jnp.log10(tarr)
    _res = _calc_halo_history_vmap(lgtarr, lgt0, lgm0, mah_lgtc, MAH_K, early, late)
    dmhdt, log_mah = _res

    return MCHaloPop(*(dmhdt, log_mah, early, late, mah_lgtc, msk_is_late))
