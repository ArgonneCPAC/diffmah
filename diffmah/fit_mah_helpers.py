"""
"""
import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from .individual_halo_assembly import _MAH_PARS, _u_rolling_plaw_vs_logt
from .individual_halo_assembly import _rolling_plaw_vs_logt
from .individual_halo_assembly import _get_u_k, _get_u_early_index, _get_u_x0
from .individual_halo_assembly import _get_params_from_u_params
from .individual_halo_assembly import _get_x0_from_early_index
from .individual_halo_assembly import _get_early_index, _get_late_index, _get_k

T_FIT_MIN = 1.5
DLOGM_CUT = 2.5


@jjit
def lge_mse_loss_fixed_x0(u_params, loss_data):
    logt, log_mah_target, logtmp, u_k = loss_data
    logmp, lge, u_dy = u_params
    early = 10 ** lge
    late = _get_late_index(u_dy, early)
    x0 = _get_x0_from_early_index(early)
    k = _get_k(u_k)

    log_mah_pred = _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, k, early, late)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


@jjit
def mse_loss_fixed_mp(u_params, loss_data):
    """Loss function to minimize for the case in which
    both x0 and M0 are held to fixed values. This function should be used
    in concert with get_loss_data_fixed_mp.

    Parameters
    ----------
    u_params : ndarray of shape (2, )

    loss_data : sequence
        See return value of get_loss_data_fixed_mp

    """
    logt, log_mah_target, logtmp, u_k, logmp = loss_data
    u_early, u_dy = u_params
    early = _get_early_index(u_early)
    late = _get_late_index(u_dy, early)
    x0 = _get_x0_from_early_index(early)
    k = _get_k(u_k)

    log_mah_pred = _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, k, early, late)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


@jjit
def mse_loss_variable_mp(u_params, loss_data):
    """Loss function to minimize for the case in which
    x0 is fixed and M0 is varied simultaneously with early_index and late_index.
    This function should be used in concert with get_loss_data_fixed_mp.

    Parameters
    ----------
    u_params : ndarray of shape (3, )

    loss_data : sequence
        See return value of get_loss_data_variable_mp

    """
    logt, log_mah_target, logtmp, u_k = loss_data
    logmp, u_early, u_dy = u_params
    early = _get_early_index(u_early)
    late = _get_late_index(u_dy, early)
    x0 = _get_x0_from_early_index(early)
    k = _get_k(u_k)

    log_mah_pred = _rolling_plaw_vs_logt(logt, logtmp, logmp, x0, k, early, late)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


@jjit
def mse_loss_variable_mp_x0(u_params, loss_data):
    """Loss function to minimize for the case in which
    both x0 and M0 are varied simultaneously with early_index and late_index.
    This function should be used in concert with get_loss_data_variable_mp_x0.

    Parameters
    ----------
    u_params : ndarray of shape (4, )

    loss_data : sequence
        See return value of get_loss_data_variable_mp_x0

    """
    logmp, u_x0, mah_u_early, mah_dy = u_params
    logt, log_mah_target, logtmp, u_k = loss_data
    log_mah_pred = _u_rolling_plaw_vs_logt(
        logt, logtmp, logmp, u_x0, u_k, mah_u_early, mah_dy
    )
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


def get_loss_data_variable_mp_x0(
    t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut=DLOGM_CUT, t_fit_min=T_FIT_MIN
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

    tmp : float
        First time the halo attains its peak mass in Gyr

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
            Base-10 log of tmp

        u_k_fixed : float
            Fixed value of the bounded diffmah parameter k

    """
    logt_target, log_mah_target = _get_target_data(
        t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut, t_fit_min
    )
    logmp_init = log_mah_sim[-1]
    mah_u_early_init = _get_u_early_index(_MAH_PARS["mah_early"])
    u_dy_init = 0.0
    u_x0_init = _get_u_x0(_MAH_PARS["mah_x0"])
    p_init = np.array((logmp_init, u_x0_init, mah_u_early_init, u_dy_init)).astype("f4")

    u_k_fixed = _get_u_k(_MAH_PARS["mah_k"])
    logtmp = np.log10(tmp)

    loss_data = (logt_target, log_mah_target, logtmp, u_k_fixed)
    return p_init, loss_data


def get_loss_data_lge_fixed_x0(
    t_sim,
    log_mah_sim,
    tmp,
    lgm_min,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    dt_fit_beyond_tmpeak=0,
):
    logt_target, log_mah_target = _get_target_data(
        t_sim,
        log_mah_sim,
        tmp,
        lgm_min,
        dlogm_cut,
        t_fit_min,
        dt_fit_beyond_tmpeak=dt_fit_beyond_tmpeak,
    )
    logmp_init = log_mah_sim[-1]
    lge_init = np.log10(_MAH_PARS["mah_early"])
    u_dy_init = 0.0
    p_init = np.array((logmp_init, lge_init, u_dy_init)).astype("f4")

    u_k_fixed = _get_u_k(_MAH_PARS["mah_k"])
    logtmp = np.log10(tmp)

    loss_data = (logt_target, log_mah_target, logtmp, u_k_fixed)
    return p_init, loss_data


def get_loss_data_variable_mp(
    t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut=DLOGM_CUT, t_fit_min=T_FIT_MIN
):
    """Retrieve the target data passed to the optimizer when fitting the halo MAH
    model for the case in which M0 is varied in addition to the early- and late-time
    power-law indices, but x0 and is held fixed to a value determined by early_index.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr

    log_mah_sim : ndarray of shape (nt, )
        Base-10 log of halo mass in Msun

    tmp : float
        First time the halo attains its peak mass in Gyr

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
    p_init : ndarray of shape (3, )
        Initial guess for the unbounded parameters used to fit the MAH.
        In this version of the fitter, the early-time and late-time power-law
        indices are varied, and the value of halo mass at the input peak time
        is also varied, so that logmp_fit represents an overall normalization of the
        MAH, and may differ from the simulated value at tmpeak.
        The x0 parameter is held fixed to a value that is analytically determined by
        the value of the early-time index. So in this version we have
        p_init = (logmp_init, mah_u_early_init, u_dy_init)

    loss_data : sequence
        logt_target : ndarray of shape (n_target, )
            Base-10 log of cosmic time of the simulation snapshots used to define
            the target MAH

        log_mah_target : ndarray of shape (n_target, )
            Base-10 log of halo mass in Msun of the target MAH

        logtmp : float
            Base-10 log of the cosmic time in Gyr that the simulated halo first
            attains its peak mass

        u_k_fixed : float
            The diffmah parameter `k` controls how quickly the halo transitions
            from fast- to slow-accretion regimes. The value of u_k_fixed is
            the unbounded version of the transition speed parameter
            and will be held fixed to this value while running the fitter.

    """
    logt_target, log_mah_target = _get_target_data(
        t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut, t_fit_min
    )
    logmp_init = log_mah_sim[-1]
    mah_u_early_init = _get_u_early_index(_MAH_PARS["mah_early"])
    u_dy_init = 0.0
    p_init = np.array((logmp_init, mah_u_early_init, u_dy_init)).astype("f4")

    u_k_fixed = _get_u_k(_MAH_PARS["mah_k"])
    logtmp = np.log10(tmp)

    loss_data = (logt_target, log_mah_target, logtmp, u_k_fixed)
    return p_init, loss_data


def get_loss_data_fixed_mp(
    t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut=DLOGM_CUT, t_fit_min=T_FIT_MIN
):
    """Retrieve the target data passed to the optimizer when fitting the halo MAH
    model for the case in which both x0 and M0 are held to fixed values.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr

    log_mah_sim : ndarray of shape (nt, )
        Base-10 log of halo mass in Msun

    tmp : float
        First time the halo attains its peak mass in Gyr

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
    p_init : ndarray of shape (2, )
        Initial guess for the unbounded parameters used to fit the MAH.
        In this version of the fitter, only the early-time and late-time power-law
        indices are varied. The M0 parameter is held fixed to the halo of
        the simulated MAH at tmpeak, the first time the halo attains this value.
        The x0 parameter is held fixed to a value that is analytically determined by
        the value of the early-time index.

    loss_data : sequence
        logt_target : ndarray of shape (n_target, )
            Base-10 log of cosmic time of the simulation snapshots used to define
            the target MAH

        log_mah_target : ndarray of shape (n_target, )
            Base-10 log of halo mass in Msun of the target MAH

        logtmp : float
            Base-10 log of the cosmic time in Gyr that the simulated halo first
            attains its peak mass

        u_k_fixed : float
            The diffmah parameter `k` controls how quickly the halo transitions
            from fast- to slow-accretion regimes. The value of u_k_fixed is
            the unbounded version of the transition speed parameter
            and will be held fixed to this value while running the fitter.

        logmp_fixed : float
            Peak mass of the simulated MAH. In this version of the fitter,
            the value of the model MAH will be held fixed to logmp_fixed

    """
    logt_target, log_mah_target = _get_target_data(
        t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut, t_fit_min
    )
    logmp_fixed = log_mah_sim[-1]
    mah_u_early_init = _get_u_early_index(_MAH_PARS["mah_early"])
    u_dy_init = 0.0
    p_init = np.array((mah_u_early_init, u_dy_init)).astype("f4")

    u_k_fixed = _get_u_k(_MAH_PARS["mah_k"])
    logtmp = np.log10(tmp)

    loss_data = (logt_target, log_mah_target, logtmp, u_k_fixed, logmp_fixed)
    return p_init, loss_data


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


def _get_target_data(
    t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut, t_fit_min, dt_fit_beyond_tmpeak=0
):
    """Retrieve the target values of the halo MAH used to fit the model.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr

    log_mah_sim : ndarray of shape (nt, )
        Base-10 log of halo mass in Msun

    tmp : float
        First time the halo attains its peak mass in Gyr

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
    msk &= t_sim <= tmp + dt_fit_beyond_tmpeak

    logt_target = np.log10(t_sim)[msk]
    log_mah_target = np.maximum.accumulate(log_mah_sim[msk])
    return logt_target, log_mah_target


def get_outline_variable_mp_x0(halo_id, loss_data, p_best, loss_best):
    """Return the string storing fitting results that will be written to disk"""
    logmp_fit, u_x0, u_early, u_dy = p_best
    logtmp, u_k = loss_data[-2:]
    tmp = 10 ** logtmp
    x0, k, early, late = _get_params_from_u_params(u_x0, u_k, u_early, u_dy)
    _d = np.array((logmp_fit, x0, k, early, late)).astype("f4")
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_outline_lge_fixed_x0(halo_id, loss_data, p_best, loss_best):
    """Return the string storing fitting results that will be written to disk"""
    logmp_fit, lge, u_dy = p_best
    logtmp, u_k = loss_data[-2:]
    tmp = 10 ** logtmp
    early = 10 ** lge
    late = _get_late_index(u_dy, early)
    x0 = _get_x0_from_early_index(early)
    k = _get_k(u_k)
    _d = np.array((logmp_fit, x0, k, early, late)).astype("f4")
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_outline_variable_mp(halo_id, loss_data, p_best, loss_best):
    """Return the string storing fitting results that will be written to disk"""
    logmp_fit, u_early, u_dy = p_best
    logtmp, u_k = loss_data[-2:]
    tmp = 10 ** logtmp
    early = _get_early_index(u_early)
    late = _get_late_index(u_dy, early)
    x0 = _get_x0_from_early_index(early)
    k = _get_k(u_k)
    _d = np.array((logmp_fit, x0, k, early, late)).astype("f4")
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_outline_fixed_mp(halo_id, loss_data, p_best, loss_best):
    """Return the string storing fitting results that will be written to disk"""
    u_early, u_dy = p_best
    logtmp, u_k, logmp_fixed = loss_data[-3:]
    tmp = 10 ** logtmp
    early = _get_early_index(u_early)
    late = _get_late_index(u_dy, early)
    x0 = _get_x0_from_early_index(early)
    k = _get_k(u_k)
    _d = np.array((logmp_fixed, x0, k, early, late)).astype("f4")
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_outline_bad_fit(halo_id, lgmp_sim, tmp):
    x0, k, early, late = -1.0, -1.0, -1.0, -1.0
    _d = np.array((lgmp_sim, x0, k, early, late)).astype("f4")
    loss_best = -1.0
    data_out = (halo_id, *_d, tmp, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.4e}".format(x) for x in data_out[1:]])
    return out + "\n"


def _get_header():
    return "# halo_id logmp_fit mah_x0 mah_k early_index late_index tmpeak loss\n"
