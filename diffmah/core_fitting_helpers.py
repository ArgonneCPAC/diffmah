"""
"""

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from .diffmah_tq import (
    DEFAULT_MAH_PARAMS,
    DiffmahParams,
    DiffmahUParams,
    _log_mah_kern_u_params,
    get_bounded_mah_params,
    get_unbounded_mah_params,
)

DLOGM_CUT = 2.5
T_FIT_MIN = 1.0


@jjit
def compute_indx_t_q_singlehalo(log_mah_table):
    logm0 = log_mah_table[-1]
    indx_t_q = jnp.argmax(log_mah_table == logm0)
    return indx_t_q


compute_indx_t_q_halopop = jjit(vmap(compute_indx_t_q_singlehalo, in_axes=(0,)))


def get_target_data(t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min):
    logm0_sim = log_mah_sim[-1]

    msk = log_mah_sim > (logm0_sim - dlogm_cut)
    msk &= log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min

    logt_target = np.log10(t_sim)[msk]
    log_mah_target = np.maximum.accumulate(log_mah_sim[msk])
    return logt_target, log_mah_target


def get_loss_data(
    t_sim,
    log_mah_sim,
    lgm_min,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
):
    logt_target, log_mah_target = get_target_data(
        t_sim,
        log_mah_sim,
        lgm_min,
        dlogm_cut,
        t_fit_min,
    )
    t_target = 10**logt_target

    logt0 = np.log10(t_sim[-1])
    indx_t_q = compute_indx_t_q_singlehalo(log_mah_sim)
    t_q = t_sim[indx_t_q]

    p_init = np.array(DEFAULT_MAH_PARAMS).astype("f4")
    p_init[0] = log_mah_sim[indx_t_q]
    p_init = DiffmahParams(*p_init)
    u_p_init = get_unbounded_mah_params(p_init)

    loss_data = (t_target, log_mah_target, t_q, logt0)
    return u_p_init, loss_data


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def log_mah_loss_uparams(u_params, loss_data):
    """MSE loss function for fitting individual halo growth."""
    t_target, log_mah_target, t_q, logt0 = loss_data

    u_params = DiffmahUParams(*u_params)
    log_mah_pred = _log_mah_kern_u_params(u_params, t_target, t_q, logt0)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


loss_and_grads_kern = jjit(value_and_grad(log_mah_loss_uparams))


def get_outline_bad_fit(halo_id, loss_data, npts_mah, algo):
    logm0, logtc, early, late = -1.0, -1.0, -1.0, -1.0
    t_q = loss_data[2]
    loss_best = -1.0
    _floats = (logm0, logtc, early, late, t_q, loss_best)
    out_list = ["{:.5e}".format(float(x)) for x in _floats]
    out_list = [str(x) for x in out_list]
    out_list = [str(halo_id), *out_list, str(npts_mah), str(algo)]
    outline = " ".join(out_list) + "\n"
    return outline


def get_outline(halo_id, loss_data, u_p_best, loss_best, npts_mah, algo):
    """Return the string storing fitting results that will be written to disk"""
    t_q = loss_data[2]
    p_best = get_bounded_mah_params(DiffmahUParams(*u_p_best))
    logm0, logtc, early, late = p_best
    _floats = (logm0, logtc, early, late, t_q, loss_best)
    out_list = ["{:.5e}".format(float(x)) for x in _floats]
    out_list = [str(x) for x in out_list]
    out_list = [str(halo_id), *out_list, str(npts_mah), str(algo)]
    outline = " ".join(out_list) + "\n"
    return outline
