"""
"""
from copy import deepcopy
import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from diffmah.individual_halo_history import DEFAULT_MAH_PARAMS
from diffmah.galaxy_quenching import DEFAULT_Q_PARAMS
from diffmah.individual_galaxy_history import DEFAULT_PARAMS as DEFAULT_MS_SFR_PARAMS
from diffmah.individual_galaxy_history import _sm_history, _integrate_sfr
from diffmah.utils import get_dt_array
from astropy.cosmology import Planck15
from jax import ops as jops

_ALL_PARAMS = deepcopy(DEFAULT_MS_SFR_PARAMS)
_ALL_PARAMS.update(DEFAULT_Q_PARAMS)
_ALL_DEFAULT_PARAMS = np.array(list(_ALL_PARAMS.values()))


@jjit
def _get_updated_params(varied_params, default_params, indx_varied_params):
    new_params = default_params
    for indx, val in zip(indx_varied_params, varied_params):
        new_params = jops.index_update(new_params, indx, val)
    return new_params


def _get_indx_varied_params(all_param_names, varied_param_names):
    return [all_param_names.index(key) for key in varied_param_names]


@jjit
def _mstar_history_kern(params, lgt, z, logmp, mah_x0, mah_k, lge, u_dy, logtmp, dt):
    ms_params = params[0:-3]
    q_params = params[-3:]

    _a = lgt, z, logtmp, logmp, mah_x0, mah_k, lge, u_dy, ms_params, q_params, dt
    dmhdt, log_mah, main_sequence_sfr, sfr, mstar = _sm_history(*_a)
    return dmhdt, log_mah, main_sequence_sfr, sfr, mstar


_a = (0, None, None, 0, 0, 0, 0, 0, 0, None)
mstar_history_halopop = jjit(jvmap(_mstar_history_kern, in_axes=_a))


@jjit
def _mstar_history_wrapper(params, data):
    lgt, z, mah_x0, mah_k, logtmp, dt = data[0:6]
    mah_target, indx_mah_pred, mstar_target, indx_sm_pred, sfh_p = data[6:11]
    mah_p = data[-1]
    logmp, lge, u_dy = mah_p
    mstar_history_kern_data = lgt, z, logmp, mah_x0, mah_k, lge, u_dy, logtmp, dt

    return _mstar_history_kern(params, *mstar_history_kern_data)


@jjit
def _mstar_loss(params, data):
    mah_target, indx_mah_pred, mstar_target, indx_sm_pred, sfh_p = data[6:11]
    dmhdt, log_mah, main_sequence_sfr, sfr, mstar = _mstar_history_wrapper(params, data)
    mstar_pred = mstar[indx_sm_pred]
    loss_sm = _mse(jnp.log10(mstar_pred), jnp.log10(mstar_target), 1.0)
    return loss_sm


@jjit
def _mstar_loss_wrapper(varied_params, wrapper_data):
    mstar_loss_data, default_params, indx_varied = wrapper_data
    params = _get_updated_params(varied_params, default_params, indx_varied)
    return _mstar_loss(params, mstar_loss_data)


def get_ssfr_loss_args(mah_params, mah_loss_data):
    lgt, z, mah_x0, mah_k, logtmp, dt = mah_loss_data[0:6]
    mah_target, indx_mah_pred, mstar_target, indx_sm_pred, sfh_p = mah_loss_data[6:11]
    ssfr_target, indx_ssfr_pred, log_ssfr_clip = mah_loss_data[11:14]
    p_init = np.copy(sfh_p)
    ssfr_loss_data = (*mah_loss_data, mah_params)
    return p_init, ssfr_loss_data


def get_sfh_loss_args(mah_params, mah_loss_data):
    lgt, z, mah_x0, mah_k, logtmp, dt = mah_loss_data[0:6]
    mah_target, indx_mah_pred, mstar_target, indx_sm_pred, sfh_p = mah_loss_data[6:11]
    p_init = np.copy(sfh_p)
    sfh_loss_data = (*mah_loss_data, mah_params)
    return p_init, sfh_loss_data


def get_mstar_loss_wrapper_args(
    mah_params,
    mah_loss_data,
    varied_param_names=(
        "lgmc_ylo",
        "lgmc_yhi",
        "lgnc_ylo",
        "lgnc_yhi",
        "lgbd_ylo",
        "lgbd_yhi",
        "lgbc",
        "u_lg_qt",
        "u_lg_qs",
        "u_lg_dq",
    ),
):
    all_params_init, sfh_loss_data = get_sfh_loss_args(mah_params, mah_loss_data)
    all_param_names = list(_ALL_PARAMS.keys())
    varied_p_init = np.array([_ALL_PARAMS[key] for key in varied_param_names])
    indx_varied = _get_indx_varied_params(all_param_names, varied_param_names)
    wrapper_data = (sfh_loss_data, all_params_init, indx_varied)
    return varied_p_init, wrapper_data


@jjit
def _ssfr_loss(params, data):
    (
        mah_target,
        indx_mah_pred,
        mstar_target,
        indx_sm_pred,
        sfh_p,
        log_ssfr_target,
        indx_ssfr_pred,
        log_ssfr_clip,
        mah_p,
    ) = data[6:15]
    _x = _mstar_history_wrapper(params, data)
    dmhdt, log_mah, main_sequence_sfr, sfr, mstar = _x
    mstar_pred = mstar[indx_sm_pred]
    ssfr_pred = sfr[indx_ssfr_pred] / mstar[indx_ssfr_pred]
    ssfr_clip = 10 ** log_ssfr_clip
    ssfr_pred = jnp.where(ssfr_pred < ssfr_clip, ssfr_clip, ssfr_pred)

    loss_sm = _mse(jnp.log10(mstar_pred), jnp.log10(mstar_target), 1)
    loss_ssfr = _mse(jnp.log10(ssfr_pred), log_ssfr_target, 1)
    return loss_sm + loss_ssfr / 4


def get_mah_loss_args(
    t_sim,
    log_mah_sim,
    sfr_sim,
    tmp,
    t_min_smh=1,
    t_min_ssfr=5,
    t_min_mah=1,
    nt_fit=50,
    log_mah_min=9,
    log_sm_min=5,
    dlog_sm=3,
    log_ssfr_clip=-11,
    fstar_taus=(0.5,),
    mah_x0=DEFAULT_MAH_PARAMS["mah_x0"],
    mah_k=DEFAULT_MAH_PARAMS["mah_k"],
):
    today = t_sim[-1]
    logmp = log_mah_sim[-1]
    lge, u_dy = 0.5, 0.5
    p_init = np.array((logmp, lge, u_dy)).astype("f4")
    logtmp = np.log10(tmp)

    msk_t_mah = t_sim > t_min_mah
    msk_t_mah &= t_sim <= tmp
    msk_log_mah = log_mah_sim > log_mah_min
    msk_mah = msk_t_mah & msk_log_mah
    log_mah_table = log_mah_sim[msk_mah]
    log_mah_t_table = t_sim[msk_mah]
    lgt_mah_target = np.log10(np.linspace(log_mah_t_table.min(), today - 1e-3, nt_fit))
    log_mah_target = np.interp(lgt_mah_target, np.log10(log_mah_t_table), log_mah_table)
    mah_target = 10.0 ** log_mah_target

    dt_sim = get_dt_array(t_sim)
    smh_sim = _integrate_sfr(sfr_sim, dt_sim)

    msk_smh = t_sim > t_min_smh
    msk_smh &= smh_sim > 10 ** log_sm_min
    msk_smh &= smh_sim > np.max(smh_sim) / 10 ** dlog_sm
    log_smh_table = np.log10(smh_sim[msk_smh])
    log_smh_t_table = t_sim[msk_smh]
    lgt_smh_target = np.log10(np.linspace(log_smh_t_table.min(), today - 1e-3, nt_fit))
    log_smh_target = np.interp(lgt_smh_target, np.log10(log_smh_t_table), log_smh_table)
    mstar_target = 10 ** log_smh_target

    msk_ssfr = t_sim > t_min_ssfr
    msk_ssfr &= smh_sim > 10 ** log_sm_min
    msk_ssfr &= smh_sim > np.max(smh_sim) / 10 ** dlog_sm
    log_ssfr_table = np.zeros_like(sfr_sim) + log_ssfr_clip
    log_ssfr_table[msk_ssfr] = np.log10(sfr_sim[msk_ssfr]) - np.log10(smh_sim[msk_ssfr])
    log_ssfr_table = log_ssfr_table[msk_smh]
    log_ssfr_t_table = t_sim[msk_smh]  # include points sSFR=-11 in the target

    lgt_ssfr_target = np.log10(
        np.linspace(log_ssfr_t_table.min(), today - 1e-3, nt_fit)
    )
    log_ssfr_target = np.interp(
        lgt_ssfr_target, np.log10(log_ssfr_t_table), log_ssfr_table
    )
    log_ssfr_target = np.where(
        log_ssfr_target < log_ssfr_clip, log_ssfr_clip, log_ssfr_target
    )
    t_table = np.linspace(0.5, today, 300)
    z_table = redshift_from_time(t_table)
    logt_table = np.log10(t_table)
    dt_table = get_dt_array(t_table)
    t_mah_target = 10 ** lgt_mah_target
    t_smh_target = 10 ** lgt_smh_target
    t_ssfr_target = 10 ** lgt_ssfr_target
    indx_mah_pred = np.searchsorted(t_table, t_mah_target)
    indx_sm_pred = np.searchsorted(t_table, t_smh_target)
    indx_ssfr_pred = np.searchsorted(t_table, t_ssfr_target)

    pred_data = logt_table, z_table, mah_x0, mah_k, logtmp, dt_table
    msp = np.array([*DEFAULT_MS_SFR_PARAMS.values()])
    qp = np.array([*DEFAULT_Q_PARAMS.values()])
    default_sfh_params = np.concatenate((msp, qp)).astype("f4")

    target_data = mah_target, indx_mah_pred, mstar_target, indx_sm_pred
    loss_data = (
        *pred_data,
        *target_data,
        default_sfh_params,
        log_ssfr_target,
        indx_ssfr_pred,
        log_ssfr_clip,
    )
    return p_init, loss_data


def redshift_from_time(time, cosmo=Planck15):
    """Inverse of astropy function age from redshift."""
    z_table = (10 ** np.linspace(2, -0.999, 500)) - 1.0
    lgt_table = np.log10(cosmo.age(z_table).value)

    lgopz_table = np.log10(z_table + 1.0)
    lgopz = np.interp(np.log10(time), lgt_table, lgopz_table)
    opz = 10 ** lgopz
    z = opz - 1.0
    return z


@jjit
def _mah_loss(params, data):
    lgt, z, mah_x0, mah_k, logtmp, dt = data[0:6]
    mah_target, indx_mah_pred, mstar_target, indx_sm_pred, sfh_p = data[6:11]
    logmp, lge, u_dy = params
    _a = lgt, z, logmp, mah_x0, mah_k, lge, u_dy, logtmp, dt
    dmhdt, log_mah, main_sequence_sfr, sfr, mstar = _mstar_history_kern(sfh_p, *_a)
    mah_pred = 10 ** log_mah[indx_mah_pred]
    loss_mah = _mse(mah_pred, mah_target, mah_target)
    return loss_mah


@jjit
def _mse(pred, obs, err):
    d = (pred - obs) / err
    return jnp.mean(d * d)


def get_outline(halo_id, mah_fit_data, sfh_fit_data, mah_loss_data, sfh_loss_data):
    """Return the string storing fitting results that will be written to disk"""
    mah_p, mah_loss = mah_fit_data[0:2]
    sfh_p, sfh_loss = sfh_fit_data[0:2]
    mah_flag, sfh_flag = mah_fit_data[-1], sfh_fit_data[-1]

    default_params, indx_varied = sfh_loss_data[-2:]
    all_sfh_params = _get_updated_params(sfh_p, default_params, indx_varied)

    mah_x0, mah_k, logtmp = mah_loss_data[2:5]
    tmp = 10 ** logtmp
    logmp, lge, u_dy = mah_p
    mah_params = (logmp, mah_x0, mah_k, lge, u_dy)

    data_out = (
        halo_id,
        *mah_params,
        tmp,
        *all_sfh_params,
        float(mah_loss),
        float(sfh_loss),
        mah_flag,
        sfh_flag,
    )

    delim = " "
    data_string = delim.join(["{:.3e}".format(x) for x in data_out[1:-2]])
    data_string = str(halo_id) + delim + data_string
    flag_string = delim.join([str(mah_flag), str(sfh_flag)])
    data_string = data_string + delim + flag_string + "\n"
    return data_string


def get_header():
    mah_keys = ("logmp", "mah_x0", "mah_k", "mah_lge", "mah_u_dy", "tmp")
    sfr_keys = ["u_" + key for key in DEFAULT_MS_SFR_PARAMS.keys()]
    q_keys = tuple(DEFAULT_Q_PARAMS.keys())
    final_keys = ("mah_loss", "sfh_loss", "mah_fit_terminates", "sfh_fit_terminates")
    colnames = ("halo_id", *mah_keys, *sfr_keys, *q_keys, *final_keys)
    header = "# " + " ".join(colnames) + "\n"
    return header
