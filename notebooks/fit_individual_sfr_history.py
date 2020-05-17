"""
"""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from jax import jit as jax_jit
from diffmah.individual_sfr_history import _individual_log_mstar_history_jax_kern
from diffmah.individual_sfr_history import individual_sfr_history
from diffmah.individual_sfr_history import DEFAULT_SFRH_PARAMS


@jax_jit
def mse_loss(params, data):
    """
    """
    logt, dtarr, indx_t0 = data[0:3]
    _x = _individual_log_mstar_history_jax_kern(*params, logt, dtarr, indx_t0)
    _log_sfr_pred, _log_sm_pred = _x
    indx_pred = data[3]
    log_sfr_pred = _log_sfr_pred[indx_pred]
    log_sm_pred = _log_sm_pred[indx_pred]

    log_sfr_target = data[4]
    log_sm_target = data[5]
    diff_sfr = log_sfr_pred - log_sfr_target
    diff_sm = log_sm_pred - log_sm_target

    loss_sfr = jax_np.sum(diff_sfr * diff_sfr) / diff_sfr.size
    loss_sm = jax_np.sum(diff_sm * diff_sm) / diff_sm.size
    return loss_sfr + loss_sm


def get_mse_loss_data(logm0, t_table, t_fit, **model_param_dict):
    logt, dtarr, indx_t0, indx_pred = _get_time_data(t_table, t_fit)
    _x = _retrieve_target_data(logm0, indx_pred, t_table, **model_param_dict)
    log_sfr_target, log_sm_target = _x
    mse_loss_data = logt, dtarr, indx_t0, indx_pred, log_sfr_target, log_sm_target
    return mse_loss_data


def get_model_param_array(logm0, **model_param_dict):
    gen = DEFAULT_SFRH_PARAMS.items()
    s = [model_param_dict.get(key, default_val) for key, default_val in gen]
    return np.array([logm0, *s]).astype("f4")


def get_model_param_dict(model_param_array):
    logm0 = model_param_array[0]
    gen = zip(DEFAULT_SFRH_PARAMS.keys(), model_param_array[1:])
    param_dict = OrderedDict([(key, val) for key, val in gen])
    return logm0, param_dict


def _get_time_data(tarr, tobs):
    logt = np.log10(tarr)
    dtarr = np.zeros_like(tarr) + np.diff(tarr).mean()
    indx_t0 = np.argmin(np.abs(logt - 1.141))
    indx_pred = np.array(list(np.argmin(np.abs(tarr - t)) for t in tobs))
    return logt, dtarr, indx_t0, indx_pred


def _retrieve_target_data(logm0, indx_pred, t_table, **kwargs):
    _log_sfr_target, _log_sm_target = individual_sfr_history(logm0, t_table, **kwargs)
    return _log_sfr_target[indx_pred], _log_sm_target[indx_pred]
