"""Helper functions used by the script to fit individual halo histories."""
import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad
from .individual_halo_assembly import DEFAULT_MAH_PARAMS
from .individual_halo_assembly import _u_rolling_plaw_vs_logt, _get_early_late

T_FIT_MIN = 1.0
DLOGM_CUT = 2.5


def get_outline(halo_id, loss_data, p_best, loss_best):
    """Return the string storing fitting results that will be written to disk"""
    logtc, ue, ul = p_best
    logt0, u_k, logm0 = loss_data[-3:]
    t0 = 10 ** logt0
    early, late = _get_early_late(ue, ul)
    fixed_k = DEFAULT_MAH_PARAMS["mah_k"]
    _d = np.array((logm0, logtc, fixed_k, early, late)).astype("f4")
    data_out = (halo_id, *_d, t0, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out[1:]])
    return out + "\n"


@jjit
def log_mah_mse_loss(params, loss_data):
    """MSE loss function for fitting individual halo growth."""
    logt, log_mah_target, logt0, fixed_k, logm0 = loss_data
    logtc, ue, ul = params

    log_mah_pred = _u_rolling_plaw_vs_logt(logt, logt0, logm0, logtc, fixed_k, ue, ul)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


@jjit
def log_mah_mse_loss_and_grads(params, loss_data):
    """MSE loss and grad function for fitting individual halo growth."""
    return value_and_grad(log_mah_mse_loss, argnums=0)(params, loss_data)


def get_loss_data(
    t_sim,
    log_mah_sim,
    lgm_min,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
):
    """Retrieve the target data passed to the optimizer when fitting the halo MAH
    model for the case in which M0 and logtc are both varied in addition to
    the early- and late-time power-law indices.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr

    log_mah_sim : ndarray of shape (nt, )
        Base-10 log of halo mass in Msun

    lgm_min : float
        Quantity used to place a cut on which simulated snapshots are used to
        define the target halo MAH.
        The value lgm_min is the base-10 log of the minimum halo mass in the MAH
        used as target data. Should typically correspond to a few hundred particles.

    dlogm_cut : float, optional
        Additional quantity used to place a cut on which simulated snapshots are used to
        define the target halo MAH.
        Snapshots will not be used when log_mah_sim falls below
        log_mah_sim[-1] - dlogm_cut. Default is set as global at top of module.

    t_fit_min : float, optional
        Additional quantity used to place a cut on which simulated snapshots are used to
        define the target halo MAH. The value of t_fit_min defines the minimum cosmic
        time in Gyr used to define the target MAH.
        Default is set as global at top of module.

    Returns
    -------
    p_init : ndarray of shape (4, )
        Initial guess at the unbounded value of the best-fit parameter.
        Here we have p_init = (logmp_fit, u_logtc_fit, u_early_fit, u_late_fit)

    loss_data : sequence consisting of the following data
        logt_target : ndarray of shape (nt_fit, )
            Base-10 log of times at which the halo reaches the target masses

        log_mah_target : ndarray of shape (nt_fit, )
            Base-10 log of target halo mass

        logt0 : float
            Base-10 log of present-day cosmic time (defined by t_sim[-1])

        u_k_fixed : float
            Fixed value of the bounded diffmah parameter k

    """
    logt_target, log_mah_target = get_target_data(
        t_sim,
        log_mah_sim,
        lgm_min,
        dlogm_cut,
        t_fit_min,
    )
    logmp_init = log_mah_sim[-1]
    lgtc_init, fixed_k, ue_init, ud_init = list(DEFAULT_MAH_PARAMS.values())
    p_init = np.array((lgtc_init, ue_init, ud_init)).astype("f4")

    logt0 = np.log10(t_sim[-1])

    loss_data = (logt_target, log_mah_target, logt0, fixed_k, logmp_init)
    return p_init, loss_data


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


def get_target_data(t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min):
    """Retrieve the target values of the halo MAH used to fit the model.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr

    log_mah_sim : ndarray of shape (nt, )
        Base-10 log of halo mass in Msun

    lgm_min : float
        Quantity used to place a cut on which simulated snapshots are used to
        define the target halo MAH.
        The value lgm_min is the base-10 log of the minimum halo mass in the MAH
        used as target data. Should typically correspond to a few hundred particles.

    dlogm_cut : float, optional
        Additional quantity used to place a cut on which simulated snapshots are used to
        define the target halo MAH.
        Snapshots will not be used when log_mah_sim falls below
        log_mah_sim[-1] - dlogm_cut. Default is set as global at top of module.

    t_fit_min : float, optional
        Additional quantity used to place a cut on which simulated snapshots are used to
        define the target halo MAH. The value of t_fit_min defines the minimum cosmic
        time in Gyr used to define the target MAH.
        Default is set as global at top of module.

    Returns
    -------
    logt_target : ndarray of shape (n_target, )
        Base-10 log of cosmic time of the simulation snapshots used to define
        the target MAH

    log_mah_target : ndarray of shape (n_target, )
        Base-10 log of cumulative peak halo mass of the target MAH

    """
    logmp_sim = log_mah_sim[-1]

    msk = log_mah_sim > (logmp_sim - dlogm_cut)
    msk &= log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min

    logt_target = np.log10(t_sim)[msk]
    log_mah_target = np.maximum.accumulate(log_mah_sim[msk])
    return logt_target, log_mah_target


def get_outline_bad_fit(halo_id, lgmp_sim, t0):
    logtc, k, early, late = -1.0, -1.0, -1.0, -1.0
    _d = np.array((lgmp_sim, logtc, k, early, late)).astype("f4")
    loss_best = -1.0
    data_out = (halo_id, *_d, t0, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.4e}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_header():
    return "# halo_id logmp_fit mah_logtc mah_k early_index late_index t0 loss\n"
