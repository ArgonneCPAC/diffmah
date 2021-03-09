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

T_FIT_MIN = 2.0
DLOGM_CUT = 2.0


@jjit
def mse_loss_fixed_mp(u_params, loss_data):
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


def get_loss_data_variable_mp(
    t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut=DLOGM_CUT, t_fit_min=T_FIT_MIN
):
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


def _get_target_data(t_sim, log_mah_sim, tmp, lgm_min, dlogm_cut, t_fit_min):
    logmp_sim = log_mah_sim[-1]

    msk = log_mah_sim > (logmp_sim - dlogm_cut)
    msk &= log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min
    msk &= t_sim <= tmp

    logt_target = np.log10(t_sim)[msk]
    log_mah_target = log_mah_sim[msk]
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
