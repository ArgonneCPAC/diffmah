"""Module implements diffmah_fitter for fitting MAHs with diffmah

"""

from collections import namedtuple
from copy import deepcopy
from warnings import warn

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from . import diffmah_kernels as dk
from .bfgs_wrapper import bfgs_adam_fallback

DLOGM_CUT = 2.5
T_FIT_MIN = 1.0
NPTS_FIT_MIN = 3  # Number of non-trivial points in the MAH, excluding MAH(z=0)
NOFIT_FILL = -99.0
EPSILON = 1e-7

HEADER = "# logm0 logtc early_index late_index t_peak loss n_points_per_fit fit_algo\n"  # noqa : E501
FIT_COLNAMES = HEADER[1:].strip().split()
DiffmahFitData = namedtuple("DiffmahFitData", FIT_COLNAMES)


VARIED_MAH_PDICT = deepcopy(dk.DEFAULT_MAH_PDICT)
VARIED_MAH_PDICT.pop("t_peak")
VariedDiffmahParams = namedtuple("VariedDiffmahParams", list(VARIED_MAH_PDICT.keys()))
VARIED_MAH_PARAMS = VariedDiffmahParams(*list(VARIED_MAH_PDICT.values()))
_MAH_PNAMES = list(VARIED_MAH_PDICT.keys())
_MAH_UPNAMES = ["u_" + key for key in _MAH_PNAMES]
VariedDiffmahUParams = namedtuple("VariedDiffmahUParams", _MAH_UPNAMES)


def diffmah_fitter(
    t_sim,
    mah_sim,
    lgm_min=-float("inf"),
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    nstep=200,
    n_warmup=1,
):
    """Fit simulated MAH with diffmah

    Parameters
    ----------
    t_sim : array, shape (n_t, )
        Age of the universe in Gyr

    mah_sim : array, shape (n_t, )
        Halo mass in units of Msun/h

    lgm_min : float, optional
        Minimum halo mass to use input halo data in the fitter. Default is -inf.

    dlogm_cut : float, optional
        Maximum change in log10(MAH) from z=0 mass to use input halo data in the fitter.
        Default is DLOGM_CUT.

    t_fit_min : float, optional
        Minimum time to use input halo data in the fitter. Default is T_FIT_MIN.

    nstep : int
        Number of gradient descent steps to use in fitter. Default is 200.
        Only applies to cases where BFGS fails and Adam is used.

    n_warmup : int
        Number of warmup iterations in gradient descent. Default is 1.
        Only applies to cases where BFGS fails and Adam is used.

    Returns
    -------
    p_best : namedtuple
        Best-fit diffmah parameters

    loss_best : float
        MSE loss of best-fit parameters

    skip_fit : bool
        True if the fitter was not run at all due to insufficient input MAH data

    fit_terminates : bool
        True if the fitter ran to completion

    code_used : int
        0 for BFGS, 1 for Adam, -1 if skip_fit is True

    loss_data : tuple
        t_target, log_mah_target, u_t_peak, logt0 = loss_data

        * u_t_peak is the unbounded version of the diffmah t_peak parameter
        * logt0 is base-10 log of the z=0 age of the universe in Gyr

    Notes
    -----
    Note that the input data is the MAH, not log10(MAH)

    """
    msg = "Mismatched shapes for diffmah_fitter inputs `mah_sim` and `t_sim`"
    assert mah_sim.shape == t_sim.shape, msg
    _check_for_logmah_vs_mah_mistake(mah_sim)

    u_p_init, loss_data, skip_fit = get_loss_data(
        t_sim, mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    if skip_fit:
        p_best = np.zeros(len(dk.DEFAULT_MAH_PARAMS)) + NOFIT_FILL
        p_best = dk.DEFAULT_MAH_PARAMS._make(p_best)
        loss_best = NOFIT_FILL
        fit_terminates = False
        code_used = -1
        fit_results = DiffmahFitResult(
            p_best, loss_best, skip_fit, fit_terminates, code_used, loss_data
        )
        return fit_results
    else:
        _res = bfgs_adam_fallback(
            loss_and_grads_kern, u_p_init, loss_data, nstep, n_warmup
        )
        u_p_best, loss_best, fit_terminates, code_used = _res
        u_t_peak = loss_data[2]
        u_p_best = dk.DEFAULT_MAH_U_PARAMS._make((*u_p_best, u_t_peak))
        p_best = dk.get_bounded_mah_params(u_p_best)
        fit_results = DiffmahFitResult(
            p_best, loss_best, skip_fit, fit_terminates, code_used, loss_data
        )
        return fit_results


_res_names = (
    "p_best",
    "loss_best",
    "skip_fit",
    "fit_terminates",
    "code_used",
    "loss_data",
)
DiffmahFitResult = namedtuple("DiffmahFitResult", _res_names)


def _check_for_logmah_vs_mah_mistake(mah_sim):
    issue_warning = mah_sim.max() < 1e4
    msg = "Values of input MAH are suspiciously small.\n"
    msg += "Double-check that you have not input log_mah"
    if issue_warning:
        warn(msg)


def write_collated_data(outname, fit_data_strings, chunk_arr=None):
    import h5py

    logm0 = fit_data_strings[:, 0].astype(float)
    logtc = fit_data_strings[:, 1].astype(float)
    early_index = fit_data_strings[:, 2].astype(float)
    late_index = fit_data_strings[:, 3].astype(float)
    t_peak = fit_data_strings[:, 4].astype(float)
    loss = fit_data_strings[:, 5].astype(float)
    n_points_per_fit = fit_data_strings[:, 6].astype(int)
    fit_algo = fit_data_strings[:, 7].astype(int)

    with h5py.File(outname, "w") as hdf:
        hdf["logm0"] = logm0
        hdf["logtc"] = logtc
        hdf["early_index"] = early_index
        hdf["late_index"] = late_index
        hdf["t_peak"] = t_peak
        hdf["loss"] = loss
        hdf["n_points_per_fit"] = n_points_per_fit
        hdf["fit_algo"] = fit_algo


@jjit
def compute_indx_t_peak_singlehalo(log_mah_table):
    logm0 = log_mah_table[-1]
    indx_t_peak = jnp.argmax(log_mah_table == logm0)
    return indx_t_peak


compute_indx_t_peak_halopop = jjit(vmap(compute_indx_t_peak_singlehalo, in_axes=(0,)))


def _get_target_data(t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min):
    logm0_sim = log_mah_sim[-1]

    msk = log_mah_sim > (logm0_sim - dlogm_cut)
    msk &= log_mah_sim > lgm_min
    msk &= t_sim >= t_fit_min

    logt_target = np.log10(t_sim)[msk]
    log_mah_target = np.maximum.accumulate(log_mah_sim[msk])
    return logt_target, log_mah_target


def _get_clipped_log_mah(mah, lgm_min):
    msk = mah <= 10**lgm_min
    clip = EPSILON + 10**lgm_min
    clipped_log_mah = np.log10(np.where(msk, clip, mah))
    return clipped_log_mah


def get_loss_data(
    t_sim,
    mah_sim,
    lgm_min=-float("inf"),
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    npts_min=NPTS_FIT_MIN,
):
    _check_for_logmah_vs_mah_mistake(mah_sim)

    log_mah_sim = _get_clipped_log_mah(mah_sim, lgm_min)

    logt_target, log_mah_target = _get_target_data(
        t_sim,
        log_mah_sim,
        lgm_min,
        dlogm_cut,
        t_fit_min,
    )
    t_target = 10**logt_target

    npts = _compute_non_trivial_npts_in_log_mah_target(log_mah_target)

    if npts < npts_min:
        skip_fit = True
        u_t_peak, logt0 = NOFIT_FILL, NOFIT_FILL
        loss_data = DiffmahFitLossData(t_target, log_mah_target, u_t_peak, logt0)
        u_p_init = np.zeros(len(VariedDiffmahUParams._fields)) + NOFIT_FILL
        return u_p_init, loss_data, skip_fit
    else:
        skip_fit = False

        logt0 = np.log10(t_sim[-1])

        indx_t_peak = compute_indx_t_peak_singlehalo(log_mah_sim)
        t_peak = t_sim[indx_t_peak]

        p_init = np.array(dk.DEFAULT_MAH_PARAMS).astype("f4")
        p_init[0] = log_mah_sim[indx_t_peak]
        p_init[4] = t_peak
        p_init = dk.DiffmahParams(*p_init)
        u_p_init_all = dk.get_unbounded_mah_params(p_init)
        u_t_peak = u_p_init_all.u_t_peak
        u_p_init = VariedDiffmahUParams(*u_p_init_all[:-1])

        loss_data = DiffmahFitLossData(t_target, log_mah_target, u_t_peak, logt0)
        return u_p_init, loss_data, skip_fit


_loss_names = ("t_target", "log_mah_target", "u_t_peak", "logt0")
DiffmahFitLossData = namedtuple("DiffmahFitLossData", _loss_names)


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def log_mah_loss_uparams(u_params_varied, loss_data):
    """MSE loss function for fitting individual halo growth."""
    t_target, log_mah_target, u_t_peak, logt0 = loss_data

    u_params = dk.DiffmahUParams(*u_params_varied, u_t_peak)
    log_mah_pred = dk._log_mah_kern_u_params(u_params, t_target, logt0)
    log_mah_loss = _mse(log_mah_pred, log_mah_target)
    return log_mah_loss


loss_and_grads_kern = jjit(value_and_grad(log_mah_loss_uparams))


def _compute_non_trivial_npts_in_log_mah_target(log_mah_target):
    try:
        logmp0 = log_mah_target[-1]
    except IndexError:
        logmp0 = -float("inf")

    msk_non_trivial = log_mah_target < logmp0
    npts_non_trivial_mah = msk_non_trivial.sum()
    return npts_non_trivial_mah


def get_outline(fit_results):
    """Transform return of diffmah_fitter into ASCII"""
    _floats = (*fit_results.p_best, fit_results.loss_best)
    out_list = ["{:.5e}".format(float(x)) for x in _floats]
    out_list = [str(x) for x in out_list]

    npts_mah = _compute_non_trivial_npts_in_log_mah_target(
        fit_results.loss_data.log_mah_target
    )

    out_list = [*out_list, str(npts_mah), str(fit_results.code_used)]
    outline = " ".join(out_list) + "\n"
    return outline


def _parse_outline(outline):
    """Parse ASCII data into return of diffmah_fitter"""
    outdata = outline.strip().split()
    formatter = [*[float] * 6, *[int] * 2]
    outdata = [f(x) for f, x in zip(formatter, outdata)]
    return DiffmahFitData(*outdata)
