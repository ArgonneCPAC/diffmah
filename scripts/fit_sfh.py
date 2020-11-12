"""Module stores loss functions for fitting smooth models to simulated SFHs."""
from collections import OrderedDict
import numpy as np
from jax import jit as jjit
from jax import numpy as jnp

from diffmah.halo_assembly import _individual_halo_assembly_jax_kern, _get_dt_array
from diffmah.epsilon_sfr_emerge import _sfr_history_kernel
from diffmah.epsilon_sfr_emerge import DEFAULT_PARAMS as DEFAULT_SFR_PARAMS
from diffmah.halo_assembly import DEFAULT_MAH_PARAMS

FB = 0.158
LG_FB = np.log10(FB)
DEFAULT_Q_PARAMS = OrderedDict(qt=14.0, dq=2.0)


@jjit
def moster17_loss(params, data):
    """
    """
    _x = _parse_moster17_loss_args(params, data)
    mah_args, sfh_args, indx_mah, indx_ssfr, indx_smh = _x[0:5]
    log_mah_target, log_smh_target, log_ssfr_target = _x[5:8]

    _y = _predict_targets(mah_args, sfh_args)
    log_mah_table, log_dmhdt_table, log_smh_table, log_ssfr_table = _y

    log_mah_pred = log_mah_table[indx_mah]
    log_smh_pred = log_smh_table[indx_smh]
    log_ssfr_pred = log_ssfr_table[indx_ssfr]

    loss_mah = _mse(log_mah_pred, log_mah_target)
    loss_smh = _mse(log_smh_pred, log_smh_target)
    loss_ssfr = _mse(log_ssfr_pred, log_ssfr_target)

    return loss_mah + loss_smh  # + loss_ssfr


@jjit
def _predict_targets(mah_args, sfh_args):
    log_mah_table, log_dmhdt_table = _individual_halo_assembly_jax_kern(*mah_args)
    log_smh_table, log_ssfr_table = _predict_sfr_target(*sfh_args)
    return log_mah_table, log_dmhdt_table, log_smh_table, log_ssfr_table


@jjit
def _predict_target_collection_kernel(
    logt_table,
    dt_table,
    z_table,
    log_ssfr_clip,
    logmp,
    dmhdt_x0,
    dmhdt_k,
    dmhdt_early_index,
    dmhdt_late_index,
    m_0,
    m_z,
    beta_0,
    beta_z,
    gamma_0,
    lgeps_n0,
    lgeps_nz,
    qt,
    dq,
    indx_tmp,
):
    mah_args, sfh_args = _get_mah_sfh_args(
        logt_table,
        dt_table,
        z_table,
        log_ssfr_clip,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        m_0,
        m_z,
        beta_0,
        beta_z,
        gamma_0,
        lgeps_n0,
        lgeps_nz,
        qt,
        dq,
        indx_tmp,
    )
    return _predict_targets(mah_args, sfh_args)


@jjit
def _parse_moster17_loss_args(params, data):
    logmp, dmhdt_x0, dmhdt_early_index, dmhdt_late_index = params[0:4]
    m_0, m_z, beta_0, beta_z, gamma_0, lgeps_n0, lgeps_nz = params[4:11]
    qt, dq = params[11:13]

    logt_table, dt_table, z_table, indx_tmp, indx_mah, indx_ssfr, indx_smh = data[:7]
    log_ssfr_clip, dmhdt_k, log_mah_target, log_smh_target, log_ssfr_target = data[7:12]

    targets = log_mah_target, log_smh_target, log_ssfr_target

    mah_args, sfh_args = _get_mah_sfh_args(
        logt_table,
        dt_table,
        z_table,
        log_ssfr_clip,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        m_0,
        m_z,
        beta_0,
        beta_z,
        gamma_0,
        lgeps_n0,
        lgeps_nz,
        qt,
        dq,
        indx_tmp,
    )

    ret = (mah_args, sfh_args, indx_mah, indx_ssfr, indx_smh, *targets)
    return ret


@jjit
def _get_mah_sfh_args(
    logt_table,
    dt_table,
    z_table,
    log_ssfr_clip,
    logmp,
    dmhdt_x0,
    dmhdt_k,
    dmhdt_early_index,
    dmhdt_late_index,
    m_0,
    m_z,
    beta_0,
    beta_z,
    gamma_0,
    lgeps_n0,
    lgeps_nz,
    qt,
    dq,
    indx_tmp,
):
    mah_args = (
        logt_table,
        dt_table,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )
    all_mah_params = (logmp, dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index)
    all_sfr_params = (m_0, m_z, beta_0, beta_z, gamma_0, lgeps_n0, lgeps_nz)
    q_params = (qt, dq)
    sfh_args = (
        logt_table,
        dt_table,
        z_table,
        all_mah_params,
        all_sfr_params,
        q_params,
        indx_tmp,
        log_ssfr_clip,
    )
    return mah_args, sfh_args


@jjit
def _predict_sfr_target(
    logt_table,
    dtarr_table,
    z_table,
    all_mah_params,
    all_sfr_params,
    q_params,
    indx_tmp,
    log_ssfr_clip,
):
    log_mah_table, log_dmhdt_table = _individual_halo_assembly_jax_kern(
        logt_table, dtarr_table, *all_mah_params, indx_tmp,
    )
    sfr_data = (10 ** log_mah_table, z_table, LG_FB, log_dmhdt_table)
    log_sfr_table = _sfr_history_kernel(all_sfr_params, sfr_data)
    log_sfr_q_table = _log_qfunc_kernel(10 ** logt_table, log_sfr_table, *q_params)
    log_smh_table = _calculate_cumulative_in_situ_mass(log_sfr_q_table, dtarr_table)

    _x = log_sfr_q_table - log_smh_table
    log_ssfr_table = jnp.where(_x < log_ssfr_clip, log_ssfr_clip, _x)

    return log_smh_table, log_ssfr_table


@jjit
def _calculate_cumulative_in_situ_mass(log_sfr, dtarr):
    log_smh = jnp.log10(jnp.cumsum(jnp.power(10, log_sfr)) * dtarr) + 9.0
    return log_smh


@jjit
def _log_qfunc_kernel(time, log_sfr, qt, dq):
    log_sfr_q = jnp.where(log_sfr - dq > log_sfr, log_sfr, log_sfr - dq)
    return _sigmoid(time, qt, 1, log_sfr, log_sfr_q)


@jjit
def _mse(prediction, target):
    diff = prediction - target
    return jnp.sum(diff * diff) / target.size


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


def get_moster17_loss_data(
    t_table,
    z_table,
    tmp,
    t_snaps,
    log_mah,
    smh,
    log_ssfrh,
    t_mah_min,
    t_smh_min,
    t_ssfrh_min,
    dmhdt_k,
    log_ssfr_clip,
    dlogmh_cut,
    dlogsm_cut,
):
    dt_table = _get_dt_array(t_table)
    logt_table = np.log10(t_table)
    indx_tmp = np.argmin(np.abs(tmp - t_table))

    logmah_min = log_mah[-1] - dlogmh_cut
    log_mah_msk = t_snaps > t_mah_min
    log_mah_msk &= t_snaps <= tmp
    log_mah_msk &= log_mah >= logmah_min
    t_mah = t_snaps[log_mah_msk]
    log_mah_target = log_mah[log_mah_msk]

    logmstar_z0 = np.log10(smh[-1])
    smh_msk = smh > 10 ** (logmstar_z0 - dlogsm_cut)
    smh_msk &= t_snaps > t_smh_min
    t_smh = t_snaps[smh_msk]
    log_smh_target = np.log10(smh[smh_msk])

    log_ssfrh_msk = t_snaps > t_ssfrh_min
    t_ssfrh = t_snaps[log_ssfrh_msk]
    log_ssfrh_target = log_ssfrh[log_ssfrh_msk]

    indx_mah = np.array([np.argmin(np.abs(t - t_table)) for t in t_mah]).astype("i4")
    indx_smh = np.array([np.argmin(np.abs(t - t_table)) for t in t_smh]).astype("i4")
    indx_ssfr = np.array([np.argmin(np.abs(t - t_table)) for t in t_ssfrh]).astype("i4")

    moster17_loss_data = (
        logt_table,
        dt_table,
        z_table,
        indx_tmp,
        indx_mah,
        indx_ssfr,
        indx_smh,
        log_ssfr_clip,
        dmhdt_k,
        log_mah_target,
        log_smh_target,
        log_ssfrh_target,
    )
    mah_params, sfr_params, q_params = get_default_params(log_mah[-1])
    p_init = np.concatenate((mah_params, sfr_params, q_params))
    return p_init, moster17_loss_data


def get_default_params(logmp):
    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))

    mah_keys = ("dmhdt_x0", "dmhdt_early_index", "dmhdt_late_index")
    mah_params = np.array((logmp, *[DEFAULT_MAH_PARAMS[key] for key in mah_keys]))
    sfr_params = np.array(list(DEFAULT_SFR_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    return mah_params, sfr_params, q_params


def get_outline(halo_id, tmp, loss_data, fit_data):
    """Return the string storing fitting results that will be written to disk"""
    best_fit_params = fit_data[0]
    loss = fit_data[1]
    dmhdt_k = loss_data[8]
    all_params = np.array((*best_fit_params[:2], dmhdt_k, *best_fit_params[2:]))
    data_out = (halo_id, *all_params, tmp, float(loss))
    out = str(halo_id) + " " + " ".join(["{:.3f}".format(x) for x in data_out[1:]])
    return out + "\n"
