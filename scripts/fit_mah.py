"""Module stores loss functions for fitting smooth models to simulated MAHs.
Each version of the calculation needs to define mse_loss, get_loss_data, get_outline,
in order to interface with history_fitting_script.py

In the version labeled fixed_logmp, logmp is held fixed to the simulation value,
and all 4 of the other history parameters are varied.

In fixed_k_x0, logmp is allowed to float along with the power-law indices,
with k and x0 held fixed to default values.

"""
from diffmah.halo_assembly import _get_dt_array, _individual_halo_assembly_jax_kern
from diffmah.halo_assembly import DEFAULT_MAH_PARAMS
import numpy as np

from jax import jit as jjit
from jax import numpy as jnp


@jjit
def mse_loss_fixed_k_x0(params, loss_data):
    """Parse the input params and data and compute the loss.
    In this version of the calculation, we hold dmhdt_x0 and dmhdt_k fixed,
    and we allow logmp to vary along with the early and late power-law indices.

    Parameters
    ----------
    params : ndarray, shape (n_params, )
        Parameters varied when optimizing the loss

    loss_data : sequence, as follows:
        logt_table : ndarray, shape (n_table, )
            Base-10 log of the integration table used to calculate
            cumulative mass from dMh/dt

        dt_table : ndarray, shape (n_table, )
            Linear time spacing between elements in the integration table
            used to calculate cumulative mass from dMh/dt

        x0_fixed : float
            Fixed value of the parameter dmhdt_x0

        k_fixed : float
            Fixed value of the parameter dmhdt_k

        indx_tmp : int
            Index where the t_table array is closest to the input halo tmp

        indx_pred : ndarray, shape (n_target, )
            Indices of t_table closest in time to the target predictions

        log_mah_target : ndarray, shape (n_target, )
            Base-10 log of peak halo mass in Msun

        log_dmhdt_target : ndarray, shape (n_target, )
            Base-10 log of halo mass accretion rate in Msun/yr

    Returns
    -------
    loss : float
        MSE loss of the difference between the simulated and predicted cumulative mass

    """
    logt_table, dt_table, x0_fixed, k_fixed, indx_tmp = loss_data[0:5]
    logmp, dmhdt_early_index, dmhdt_late_index = params

    args = (
        logt_table,
        dt_table,
        logmp,
        x0_fixed,
        k_fixed,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )
    log_mah_table, log_dmhdt_table = _individual_halo_assembly_jax_kern(*args)
    indx_pred = loss_data[5]
    log_mah_pred = log_mah_table[indx_pred]

    log_mah_target, log_dmhdt_target = loss_data[6:8]
    log_mah_loss = _mse(log_mah_pred, log_mah_target)

    return log_mah_loss


@jjit
def mse_loss_fixed_logmp(params, loss_data):
    """Parse the input params and data and compute the loss.
    In this version of the calculation, logmp is held fixed to the value in the sim,
    and all dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index are varied.

    Parameters
    ----------
    params : ndarray, shape (n_params, )
        Parameters varied when optimizing the loss

    loss_data : sequence, as follows:

        logt_table : ndarray, shape (n_table, )
            Base-10 log of the integration table used to calculate
            cumulative mass from dMh/dt

        dt_table : ndarray, shape (n_table, )
            Linear time spacing between elements in the integration table
            used to calculate cumulative mass from dMh/dt

        logmp : float
            Base-10 log of peak halo mass in Msun

        indx_tmp : int
            Index where the t_table array is closest to the input halo tmp

        indx_pred : ndarray, shape (n_target, )
            Indices of t_table closest in time to the target predictions

        log_mah_target : ndarray, shape (n_target, )
            Base-10 log of peak halo mass in Msun

        log_dmhdt_target : ndarray, shape (n_target, )
            Base-10 log of halo mass accretion rate in Msun/yr

    Returns
    -------
    loss : floats
        MSE loss of the difference between the simulated and predicted cumulative mass

    """
    logt_table, dt_table, logmp, indx_tmp = loss_data[0:4]
    dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index = params

    args = (
        logt_table,
        dt_table,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )
    log_mah_table, log_dmhdt_table = _individual_halo_assembly_jax_kern(*args)
    indx_pred = loss_data[4]
    log_mah_pred = log_mah_table[indx_pred]

    log_mah_target, log_dmhdt_target = loss_data[5:7]
    log_mah_loss = _mse(log_mah_pred, log_mah_target)

    return log_mah_loss


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


def get_loss_data_fixed_logmp(
    t_simulation,
    log_mah_simulated_halo,
    log_dmhdt_simulated_halo,
    tmp,
    t_table=np.logspace(-2, 1.15, 500),
    dlogm_cut=2.5,
    t_fit_min=1,
):
    """Compute the loss data for the input halo.
    In this version of the calculation, logmp is held fixed to the value in the sim,
    and all dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index are varied.

    Parameters
    ----------
    t_simulation : ndarray, shape (n_times, )
        Age of the universe in Gyr of the simulation snapshots

    log_mah_simulated_halo : ndarray, shape (n_times, )
        Base-10 log of peak halo mass in Msun

    log_dmhdt_simulated_halo : ndarray, shape (n_times, )
        Base-10 log of halo mass accretion rate in Msun/yr

    tmp : float
        Time the halo first reached its peak halo mass in Gyr

    t_table : ndarray, shape (n_table, ), optional
        Integration table used to calculate cumulative mass from dMh/dt
        Should span the range of t_simulation.

    dlogm_cut : float, optional
        Parameter used to restrict the range of the simulated MAH used in the fit.
        Snapshots for which log_mah_simulated_halo < (logmp - dlogm_cut)
        will be excluded from the fitting data

    t_fit_min : float, optional
        Parameter used to restrict the range of the simulated MAH used in the fit.
        Snapshots with t_simulation < t_fit_min will be excluded from the fitting data.

    Returns
    -------
    loss_data : sequence
        Tuple of data passed to the halo MAH fitter:

        logt_table : ndarray, shape (n_table, )
            Base-10 log of the integration table used to calculate
            cumulative mass from dMh/dt

        dt_table : ndarray, shape (n_table, )
            Linear time spacing between elements in the integration table
            used to calculate cumulative mass from dMh/dt

        logmp : float
            Base-10 log of peak halo mass in Msun

        indx_tmp : int
            Index where the t_table array is closest to the input halo tmp

        indx_pred : ndarray, shape (n_target, )
            Indices of t_table closest in time to the target predictions

        log_mah_target : ndarray, shape (n_target, )
            Base-10 log of peak halo mass in Msun

        log_dmhdt_target : ndarray, shape (n_target, )
            Base-10 log of halo mass accretion rate in Msun/yr

    p_init : ndarray, shape (n_varied_params, )
        Initial guess at the best-fit value for each parameter varied in the fit

    """
    logmp = log_mah_simulated_halo[-1]
    dt_table = _get_dt_array(t_table)
    logt_table = np.log10(t_table)

    msk = log_mah_simulated_halo > logmp - dlogm_cut
    msk &= t_simulation > t_fit_min
    log_mah_target = log_mah_simulated_halo[msk]
    log_dmhdt_target = log_dmhdt_simulated_halo[msk]
    t_target = t_simulation[msk]
    indx_pred = np.array([np.argmin(np.abs(t_table - t)) for t in t_target]).astype(
        "i4"
    )
    indx_tmp = np.argmin(np.abs(t_table - tmp))

    p_init = np.array(list(DEFAULT_MAH_PARAMS.values()))
    loss_data = (
        logt_table,
        dt_table,
        logmp,
        indx_tmp,
        indx_pred,
        log_mah_target,
        log_dmhdt_target,
    )
    return loss_data, p_init


def get_loss_data_fixed_k_x0(
    t_simulation,
    log_mah_simulated_halo,
    log_dmhdt_simulated_halo,
    tmp,
    t_table=np.logspace(-2, 1.15, 500),
    dlogm_cut=2.5,
    t_fit_min=1,
):
    """Compute the loss data for the input halo.
    In this version of the calculation, we hold dmhdt_x0 and dmhdt_k fixed,
    and we allow logmp to vary along with the early and late power-law indices.

    Parameters
    ----------
    t_simulation : ndarray, shape (n_times, )
        Age of the universe in Gyr of the simulation snapshots

    log_mah_simulated_halo : ndarray, shape (n_times, )
        Base-10 log of peak halo mass in Msun

    log_dmhdt_simulated_halo : ndarray, shape (n_times, )
        Base-10 log of halo mass accretion rate in Msun/yr

    tmp : float
        Time the halo first reached its peak halo mass in Gyr

    t_table : ndarray, shape (n_table, ), optional
        Integration table used to calculate cumulative mass from dMh/dt
        Should span the range of t_simulation.

    dlogm_cut : float, optional
        Parameter used to restrict the range of the simulated MAH used in the fit.
        Snapshots for which log_mah_simulated_halo < (logmp - dlogm_cut)
        will be excluded from the fitting data

    t_fit_min : float, optional
        Parameter used to restrict the range of the simulated MAH used in the fit.
        Snapshots with t_simulation < t_fit_min will be excluded from the fitting data.

    Returns
    -------
    loss_data : sequence, as follows:

        logt_table : ndarray, shape (n_table, )
            Base-10 log of the integration table used to calculate
            cumulative mass from dMh/dt

        dt_table : ndarray, shape (n_table, )
            Linear time spacing between elements in the integration table
            used to calculate cumulative mass from dMh/dt

        x0_fixed : float
            Fixed value of the parameter dmhdt_x0

        k_fixed : float
            Fixed value of the parameter dmhdt_k

        indx_tmp : int
            Index where the t_table array is closest to the input halo tmp

        indx_pred : ndarray, shape (n_target, )
            Indices of t_table closest in time to the target predictions

        log_mah_target : ndarray, shape (n_target, )
            Base-10 log of peak halo mass in Msun

        log_dmhdt_target : ndarray, shape (n_target, )
            Base-10 log of halo mass accretion rate in Msun/yr

    p_init : ndarray, shape (n_varied_params, )
        Initial guess at the best-fit value for each parameter varied in the fit

    """
    dt_table = _get_dt_array(t_table)
    logt_table = np.log10(t_table)

    logmp_sim = log_mah_simulated_halo[-1]

    msk = log_mah_simulated_halo > logmp_sim - dlogm_cut
    msk &= t_simulation > t_fit_min
    log_mah_target = log_mah_simulated_halo[msk]
    log_dmhdt_target = log_dmhdt_simulated_halo[msk]
    t_target = t_simulation[msk]
    indx_pred = np.array([np.argmin(np.abs(t_table - t)) for t in t_target]).astype(
        "i4"
    )
    indx_tmp = np.argmin(np.abs(t_table - tmp))

    logmp_init = np.copy(logmp_sim)
    early_init = DEFAULT_MAH_PARAMS["dmhdt_early_index"]
    late_init = DEFAULT_MAH_PARAMS["dmhdt_late_index"]
    p_init = np.array((logmp_init, early_init, late_init))

    x0_fixed = DEFAULT_MAH_PARAMS["dmhdt_x0"]
    k_fixed = DEFAULT_MAH_PARAMS["dmhdt_k"]

    loss_data = (
        logt_table,
        dt_table,
        x0_fixed,
        k_fixed,
        indx_tmp,
        indx_pred,
        log_mah_target,
        log_dmhdt_target,
    )
    return loss_data, p_init


def get_outline_fixed_logmp(halo_id, tmp, loss_data, fit_data):
    """Return the string storing fitting results that will be written to disk"""
    params, loss = fit_data[:2]
    logmp = loss_data[2]
    data_out = (halo_id, logmp, *params, tmp, float(loss))
    out = str(halo_id) + " " + " ".join(["{:.3f}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_outline_fixed_k_x0(halo_id, tmp, loss_data, fit_data):
    """Return the string storing fitting results that will be written to disk"""
    best_fit_params = fit_data[0]
    loss = fit_data[1]
    x0, k = loss_data[2:4]
    logmp = float(best_fit_params[0])
    all_params = np.array((x0, k, best_fit_params[1], best_fit_params[2]))
    data_out = (halo_id, logmp, *all_params, tmp, float(loss))
    out = str(halo_id) + " " + " ".join(["{:.3f}".format(x) for x in data_out[1:]])
    return out + "\n"
