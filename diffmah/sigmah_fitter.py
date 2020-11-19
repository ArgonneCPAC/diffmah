"""
"""
import numpy as np
from sigmah import _get_unbounded_params, _get_bounded_params, _rolling_plaw_bounded
from jax import jit as jjit
from jax import numpy as jnp


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def _rolling_plaw_loss(params, data):
    logmp, u_x0, u_ymin, u_dy = params
    logt, logtmp, k, indx_pred, log_mah_target = data
    log_mah_pred = _rolling_plaw_bounded(logt, logtmp, logmp, u_x0, k, u_ymin, u_dy)

    return _mse(log_mah_pred[indx_pred], log_mah_target)


@jjit
def _rolling_plaw_loss_fixed_x0(params, data):
    logmp, u_ymin, u_dy = params
    logt, logtmp, u_x0, k, indx_pred, log_mah_target = data
    log_mah_pred = _rolling_plaw_bounded(logt, logtmp, logmp, u_x0, k, u_ymin, u_dy)

    return _mse(log_mah_pred[indx_pred], log_mah_target)


def get_sigmah_loss_data(
    t_simulation,
    log_mah_simulated_halo,
    log_dmhdt_simulated_halo,
    tmp,
    t_table=np.logspace(-2, 1.15, 500),
    dlogm_cut=3.0,
    t_fit_min=0.5,
    min_logmp=10,
    k=4.0,
):
    logmp = log_mah_simulated_halo[-1]
    logt_table = np.log10(t_table)

    logmp_cut = max(10, logmp - dlogm_cut)
    msk = log_mah_simulated_halo > logmp_cut
    msk &= t_simulation > t_fit_min
    log_mah_target = log_mah_simulated_halo[msk]

    t_target = t_simulation[msk]
    indx_pred = np.array([np.argmin(np.abs(t_table - t)) for t in t_target]).astype(
        "i4"
    )

    p_init = np.array((logmp, 0.0, 0.0, 0.0))
    loss_data = logt_table, np.log10(tmp), k, indx_pred, log_mah_target
    return loss_data, p_init


def get_sigmah_fixed_x0_loss_data(
    t_simulation,
    log_mah_simulated_halo,
    log_dmhdt_simulated_halo,
    tmp,
    t_table=np.logspace(-2, 1.15, 500),
    dlogm_cut=3.0,
    t_fit_min=0.5,
    min_logmp=10,
    k=4.0,
    x0=-0.15,
):
    logmp = log_mah_simulated_halo[-1]
    logt_table = np.log10(t_table)

    logmp_cut = max(10, logmp - dlogm_cut)
    msk = log_mah_simulated_halo > logmp_cut
    msk &= t_simulation > t_fit_min
    log_mah_target = log_mah_simulated_halo[msk]

    t_target = t_simulation[msk]
    indx_pred = np.array([np.argmin(np.abs(t_table - t)) for t in t_target]).astype(
        "i4"
    )
    u_x0 = _get_unbounded_params(x0, 1, 1)[0]

    p_init = np.array((logmp, 0.0, 0.0))
    loss_data = logt_table, np.log10(tmp), u_x0, k, indx_pred, log_mah_target
    return loss_data, p_init


def get_outline_sigmah(halo_id, tmp, loss_data, fit_data):
    """Return the string storing fitting results that will be written to disk"""
    best_fit_uparams = fit_data[0]
    logmp, u_x0, u_early, u_dy = best_fit_uparams
    x0, early_index, late_index = _get_bounded_params(u_x0, u_early, u_dy)

    loss = max(1e-5, float(fit_data[1]))
    logloss = np.log10(loss)
    k_fixed = loss_data[2]
    all_params = np.array((logmp, x0, k_fixed, early_index, late_index))
    data_out = (halo_id, *all_params, tmp, logloss)
    out = str(halo_id) + " " + " ".join(["{:.4f}".format(x) for x in data_out[1:]])
    return out + "\n"


def get_outline_sigmah_fixed_x0(halo_id, tmp, loss_data, fit_data):
    """Return the string storing fitting results that will be written to disk"""
    best_fit_uparams = fit_data[0]
    logmp, u_early, u_dy = best_fit_uparams
    __, early_index, late_index = _get_bounded_params(0, u_early, u_dy)

    loss = max(1e-5, float(fit_data[1]))
    logloss = np.log10(loss)
    u_x0_fixed, k_fixed = loss_data[2:4]
    x0_fixed = _get_bounded_params(u_x0_fixed, 0, 0)[0]
    all_params = np.array((logmp, x0_fixed, k_fixed, early_index, late_index))
    data_out = (halo_id, *all_params, tmp, logloss)
    out = str(halo_id) + " " + " ".join(["{:.4f}".format(x) for x in data_out[1:]])
    return out + "\n"
