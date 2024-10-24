"""Utility functions for fitting MAHs with diffmah.
Data loading functions require h5py and/or haccytrees

"""

import os
from collections import namedtuple
from copy import deepcopy

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from . import diffmah_kernels as dk
from .bfgs_wrapper import bfgs_adam_fallback
from .diffmah_kernels import (
    DEFAULT_MAH_PARAMS,
    DEFAULT_MAH_PDICT,
    DiffmahParams,
    DiffmahUParams,
    _log_mah_kern_u_params,
    get_bounded_mah_params,
    get_unbounded_mah_params,
)

DLOGM_CUT = 2.5
T_FIT_MIN = 1.0
HEADER = "# halo_id logm0 logtc early_index late_index t_peak loss n_points_per_fit fit_algo\n"  # noqa : E501
DEFAULT_NCHUNKS = 50

VARIED_MAH_PDICT = deepcopy(DEFAULT_MAH_PDICT)
VARIED_MAH_PDICT.pop("t_peak")
VariedDiffmahParams = namedtuple("VariedDiffmahParams", list(VARIED_MAH_PDICT.keys()))
VARIED_MAH_PARAMS = VariedDiffmahParams(*list(VARIED_MAH_PDICT.values()))
_MAH_PNAMES = list(VARIED_MAH_PDICT.keys())
_MAH_UPNAMES = ["u_" + key for key in _MAH_PNAMES]
VariedDiffmahUParams = namedtuple("VariedDiffmahUParams", _MAH_UPNAMES)

LJ_Om = 0.310
LJ_h = 0.6766


def diffmah_fitter(
    t_sim,
    log_mah_sim,
    lgm_min,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    nstep=200,
    n_warmup=1,
):
    u_p_init, loss_data = get_loss_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    _res = bfgs_adam_fallback(loss_and_grads_kern, u_p_init, loss_data, nstep, n_warmup)
    u_p_best, loss_best, fit_terminates, code_used = _res
    u_t_peak = loss_data[2]
    u_p_best = dk.DEFAULT_MAH_U_PARAMS(*u_p_best, u_t_peak)
    p_best = dk.get_bounded_mah_params(u_p_best)
    return p_best, loss_best, fit_terminates, code_used


def write_collated_data(outname, fit_data_strings, chunk_arr=None):
    import h5py

    halo_id = fit_data_strings[:, 0].astype(int)
    logm0 = fit_data_strings[:, 1].astype(float)
    logtc = fit_data_strings[:, 2].astype(float)
    early_index = fit_data_strings[:, 3].astype(float)
    late_index = fit_data_strings[:, 4].astype(float)
    t_peak = fit_data_strings[:, 5].astype(float)
    loss = fit_data_strings[:, 6].astype(float)
    n_points_per_fit = fit_data_strings[:, 7].astype(int)
    fit_algo = fit_data_strings[:, 8].astype(int)

    with h5py.File(outname, "w") as hdf:
        hdf["halo_id"] = halo_id
        hdf["logm0"] = logm0
        hdf["logtc"] = logtc
        hdf["early_index"] = early_index
        hdf["late_index"] = late_index
        hdf["t_peak"] = t_peak
        hdf["loss"] = loss
        hdf["n_points_per_fit"] = n_points_per_fit
        hdf["fit_algo"] = fit_algo

        if chunk_arr is not None:
            hdf["chunk"] = chunk_arr


@jjit
def compute_indx_t_peak_singlehalo(log_mah_table):
    logm0 = log_mah_table[-1]
    indx_t_peak = jnp.argmax(log_mah_table == logm0)
    return indx_t_peak


compute_indx_t_peak_halopop = jjit(vmap(compute_indx_t_peak_singlehalo, in_axes=(0,)))


def get_target_data(t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min):
    logm0_sim = log_mah_sim[-1]

    msk = log_mah_sim > (logm0_sim - dlogm_cut)
    msk &= log_mah_sim > lgm_min
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
    indx_t_peak = compute_indx_t_peak_singlehalo(log_mah_sim)
    t_peak = t_sim[indx_t_peak]

    p_init = np.array(DEFAULT_MAH_PARAMS).astype("f4")
    p_init[0] = log_mah_sim[indx_t_peak]
    p_init[4] = t_peak
    p_init = DiffmahParams(*p_init)
    u_p_init_all = get_unbounded_mah_params(p_init)
    u_t_peak = u_p_init_all.u_t_peak
    u_p_init = VariedDiffmahUParams(*u_p_init_all[:-1])

    loss_data = (t_target, log_mah_target, u_t_peak, logt0)
    return u_p_init, loss_data


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def log_mah_loss_uparams(u_params_varied, loss_data):
    """MSE loss function for fitting individual halo growth."""
    t_target, log_mah_target, u_t_peak, logt0 = loss_data

    u_params = DiffmahUParams(*u_params_varied, u_t_peak)
    log_mah_pred = _log_mah_kern_u_params(u_params, t_target, logt0)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


loss_and_grads_kern = jjit(value_and_grad(log_mah_loss_uparams))


def get_outline_bad_fit(halo_id, loss_data, npts_mah, algo):
    log_mah_target = loss_data[1]
    logm0 = log_mah_target[-1]
    logtc, early, late = -1.0, -1.0, -1.0
    u_t_peak = loss_data[2]
    t_peak = dk._get_bounded_diffmah_param(u_t_peak, dk.MAH_PBOUNDS.t_peak)
    loss_best = -1.0
    _floats = (logm0, logtc, early, late, t_peak, loss_best)
    out_list = ["{:.5e}".format(float(x)) for x in _floats]
    out_list = [str(x) for x in out_list]
    out_list = [str(halo_id), *out_list, str(npts_mah), str(algo)]
    outline = " ".join(out_list) + "\n"
    return outline


def get_outline(halo_id, loss_data, u_p_best, loss_best, npts_mah, algo):
    """Return the string storing fitting results that will be written to disk"""
    u_t_peak = loss_data[2]
    p_best = get_bounded_mah_params(DiffmahUParams(*u_p_best, u_t_peak))
    logm0, logtc, early, late, t_peak = p_best
    _floats = (logm0, logtc, early, late, t_peak, loss_best)
    out_list = ["{:.5e}".format(float(x)) for x in _floats]
    out_list = [str(x) for x in out_list]
    out_list = [str(halo_id), *out_list, str(npts_mah), str(algo)]
    outline = " ".join(out_list) + "\n"
    return outline


def load_lj_mahs(fdir, subvolume, chunknum, nchunks=DEFAULT_NCHUNKS, lgmh_clip=7):
    from dsps.cosmology.flat_wcdm import age_at_z
    from haccytrees import Simulation as HACCSim
    from haccytrees import coretrees

    simulation = HACCSim.simulations["LastJourney"]
    zarr = simulation.step2z(np.array(simulation.cosmotools_steps))

    fname = os.path.join(fdir, "m000p.coreforest.%s.hdf5" % subvolume)
    forest_matrices = coretrees.corematrix_reader(
        fname,
        calculate_secondary_host_row=True,
        nchunks=nchunks,
        chunknum=chunknum,
        simulation="LastJourney",
    )

    # Clip mass at LOGMH_MIN and set log_mah==0 when below the clip
    core_mass = forest_matrices["infall_fof_halo_mass"]
    mahs = np.maximum.accumulate(core_mass, axis=1)
    mahs = np.where(mahs < 10**lgmh_clip, 1.0, mahs)
    log_mahs = np.log10(mahs)  # now log-safe thanks to clip
    tarr = age_at_z(zarr, LJ_Om, -1.0, 0.0, LJ_h)
    return zarr, tarr, log_mahs
