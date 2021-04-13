"""
"""
import numpy as np


def monte_carlo_halo_histories(
    cosmic_time,
    logmh,
    lge_arr=LGE_ARR,
    lgl_arr=LGL_ARR,
    x0_arr=X0_ARR,
    seed=0,
    **kwargs
):
    cosmic_time = np.atleast_1d(cosmic_time)
    logmh = np.atleast_1d(logmh)
    n_halos = logmh.size

    lgt = np.log10(cosmic_time)

    ran_key = jran.PRNGKey(seed)
    gal_type_key, early_key, late_key = jran.split(ran_key, 3)
    uran = jran.uniform(gal_type_key, shape=(n_halos,))
    n_late = (uran < frac_late).sum()
    n_early = n_halos - n_late

    _res = _get_mah_means_and_covs(lgt, logmh, lge_arr, lgl_arr, x0_arr, **kwargs)
    frac_late, means_early, covs_early, means_late, covs_late = _res
