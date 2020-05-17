"""
"""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from jax import jit as jax_jit
from diffmah.halo_assembly import DEFAULT_MAH_PARAMS
from diffmah.halo_assembly import individual_halo_assembly_history
from diffmah.halo_assembly import _individual_halo_assembly_jax_kern


@jax_jit
def mse_loss(params, data):
    """
    """
    logt, dtarr, indx_t0 = data[0:3]
    log_mah_table = _individual_halo_assembly_jax_kern(*params, logt, dtarr, indx_t0)[0]
    indx_pred = data[3]
    log_mah_pred = log_mah_table[indx_pred]

    log_mah_target = data[4]
    diff_log_mah = log_mah_pred - log_mah_target
    loss = jax_np.sum(diff_log_mah * diff_log_mah) / diff_log_mah.size
    return loss


def get_mse_loss_data(logm0, t_table, t_fit, **model_param_dict):
    logt, dtarr, indx_t0, indx_pred = _get_time_data(t_table, t_fit)
    log_mah_target = _retrieve_target_data(
        logm0, indx_pred, t_table, **model_param_dict
    )
    mse_loss_data = logt, dtarr, indx_t0, indx_pred, log_mah_target
    return mse_loss_data


def get_model_param_array(logm0, **model_param_dict):
    gen = DEFAULT_MAH_PARAMS.items()
    s = [model_param_dict.get(key, default_val) for key, default_val in gen]
    return np.array([logm0, *s]).astype("f4")


def get_model_param_dict(model_param_array):
    logm0 = model_param_array[0]
    gen = zip(DEFAULT_MAH_PARAMS.keys(), model_param_array[1:])
    param_dict = OrderedDict([(key, val) for key, val in gen])
    return logm0, param_dict


def _get_time_data(tarr, tobs):
    logt = np.log10(tarr)
    dtarr = np.zeros_like(tarr) + np.diff(tarr).mean()
    indx_t0 = np.argmin(np.abs(logt - 1.141))
    indx_pred = np.array(list(np.argmin(np.abs(tarr - t)) for t in tobs))
    return logt, dtarr, indx_t0, indx_pred


def _retrieve_target_data(logm0, indx_pred, t_table, **kwargs):
    _x = individual_halo_assembly_history(logm0, t_table, **kwargs)
    return _x[0][indx_pred]
