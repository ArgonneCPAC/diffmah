"""Generate Diffmah parameters for halos identified at higher redshift
"""
import numpy as np

from .halo_population_assembly import LGT0
from .individual_halo_assembly import calc_halo_history
from .monte_carlo_halo_population import mc_halo_population


def mc_diffmah_params_hiz(ran_key, t_obs, logmh, lgt0=LGT0, npop=int(1e5)):
    """Monte Carlo generator of Diffmah parameters for halos identified at t_obs.

    The returned Diffmah parameters produce a MAH that passes through logmh at t_obs.

    Parameters
    ----------
    ran_key : jax.random.PRNGKey(seed)
        jax random number key

    t_obs : float
        Age of the universe at the time the input halos attain logmh
        Cannot be greater than t0

    logmh : ndarray of shape (n_h, )
        Stores base-10 log of the mass of halos identified at t_obs

    lgt0 : float, optional
        Base-10 log of the age of the universe in Gyr
        Default value is set in diffmah and is approximately 1.14

    Returns
    -------
    logm0 : ndarray of shape (n_h, )

    lgtc : ndarray of shape (n_h, )

    early_index : ndarray of shape (n_h, )

    late_index : ndarray of shape (n_h, )


    Notes
    -----
    This function is in prototype stage and has not been validated against simulations.

    """
    t0 = 10**lgt0
    TOL = 1e-4
    if 0 <= (t_obs - t0) < TOL:
        t_obs = t0
    elif (t_obs - t0) > TOL:
        msg = "Input t_obs={0} cannot exceed t0={1}"
        assert t_obs <= t0, msg.format(t_obs, t0)

    if t_obs == t0:
        tarr = np.array((t_obs, t0))
        halopop = mc_halo_population(tarr, t0, logmh, ran_key=ran_key)
        lgm0 = logmh
        early_index = np.array(halopop.early_index)
        late_index = np.array(halopop.late_index)
        lgtc = np.array(halopop.lgtc)
    else:
        lgm0_guess = _guess_logmp_z0(t_obs, logmh, lgt0, npop)

        tarr = np.array((t_obs, t0))
        halopop = mc_halo_population(tarr, t0, lgm0_guess, ran_key=ran_key)

        early_index = np.array(halopop.early_index)
        late_index = np.array(halopop.late_index)
        lgtc = np.array(halopop.lgtc)

        res = calc_halo_history(
            tarr, t0, lgm0_guess, 10**lgtc, early_index, late_index
        )
        lgmh_guess = res[1][:, 0]
        lgdiff = lgmh_guess - logmh
        lgm0 = lgm0_guess - lgdiff

    return lgm0, lgtc, early_index, late_index


def _guess_logmp_z0(t_obs, logmh, lgt0, npop):
    tarr = np.array((t_obs, 10**lgt0))
    lgm0arr = np.linspace(logmh.min(), logmh.max() + 3, npop)

    halopop = mc_halo_population(tarr, 10**LGT0, lgm0arr)
    log_mah = np.array(halopop.log_mah)

    logmp_hiz_table = np.linspace(logmh.min() - 0.1, logmh.max() + 0.1, 50)
    dlgmp = np.diff(logmp_hiz_table)[0]

    masks = [np.abs(log_mah[:, 0] - lgm) < dlgmp for lgm in logmp_hiz_table]
    ns = [msk.sum() for msk in masks]
    logmp_hiz_table_filtered = [lgm for n, lgm in zip(ns, logmp_hiz_table) if n > 0]
    masks_filtered = [msk for n, msk in zip(ns, masks) if n > 0]

    logmp_z0_table = [np.median(log_mah[:, 1][msk]) for msk in masks_filtered]

    logmp_z0 = np.interp(logmh, logmp_hiz_table_filtered, logmp_z0_table)

    return logmp_z0
