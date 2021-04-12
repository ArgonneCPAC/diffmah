"""
"""
import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from .individual_halo_assembly import DEFAULT_MAH_PARAMS
from .individual_halo_assembly import _rolling_plaw_vs_logt

T_FIT_MIN = 1.0
DLOGM_CUT = 2.5


def get_outline(halo_id, loss_data, p_best, loss_best):
    """Return the string storing fitting results that will be written to disk"""
    logmp_fit, x0, lge, lgl = p_best
    logtmp, u_k = loss_data[-2:]
    tmp = 10 ** logtmp
    early = 10 ** lge
    late = 10 ** lgl
    fixed_k = DEFAULT_MAH_PARAMS["mah_k"]
    _d = np.array((logmp_fit, x0, fixed_k, early, late)).astype("f4")
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out[1:]])
    return out + "\n"


@jjit
def log_mah_mse_loss(lge_params, loss_data):
    """"""
    logt, log_mah_target, logtmp, fixed_k = loss_data
    logmp, x0, lge, lgl = lge_params
    early, late = 10 ** lge, 10 ** lgl

    log_mah_pred = _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, fixed_k, early, late)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


def get_loss_data(
    t_sim,
    log_mah_sim,
    lgm_min,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
):
    """Retrieve the target data passed to the optimizer when fitting the halo MAH
    model for the case in which M0 and x0 are both varied in addition to
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
        Here we have p_init = (logmp_fit, u_x0_fit, u_early_fit, u_late_fit)

    loss_data : sequence consisting of the following data
        logt_target : ndarray of shape (nt_fit, )
            Base-10 log of times at which the halo reaches the target masses

        log_mah_target : ndarray of shape (nt_fit, )
            Base-10 log of target halo mass

        logtmp : float
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
    lge_init = np.log10(DEFAULT_MAH_PARAMS["mah_early"])
    lgl_init = np.log10(DEFAULT_MAH_PARAMS["mah_late"])
    x0_init = DEFAULT_MAH_PARAMS["mah_x0"]
    fixed_k = DEFAULT_MAH_PARAMS["mah_k"]
    p_init = np.array((logmp_init, x0_init, lge_init, lgl_init)).astype("f4")

    logtmp = np.log10(t_sim[-1])

    loss_data = (logt_target, log_mah_target, logtmp, fixed_k)
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
        Base-10 log of halo mass in Msun of the target MAH

    """
    logmp_sim = log_mah_sim[-1]

    msk = log_mah_sim > (logmp_sim - dlogm_cut)
    msk &= log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min

    logt_target = np.log10(t_sim)[msk]
    log_mah_target = np.maximum.accumulate(log_mah_sim[msk])
    return logt_target, log_mah_target


def get_outline_bad_fit(halo_id, lgmp_sim, tmp):
    x0, k, early, late = -1.0, -1.0, -1.0, -1.0
    _d = np.array((lgmp_sim, x0, k, early, late)).astype("f4")
    loss_best = -1.0
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.4e}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_header():
    return "# halo_id logmp_fit mah_x0 mah_k early_index late_index tmpeak loss\n"
