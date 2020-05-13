"""
"""
import argparse
from time import time
from collections import OrderedDict
import numpy as np
from jax import numpy as jax_np
from jax import value_and_grad
from jax.experimental import optimizers as jax_opt
from diffmah.mean_sfr_history import _mean_log_mstar_history_jax_kern
from diffmah.mean_sfr_history import get_mean_galaxy_history
from diffmah.halo_assembly import MEAN_MAH_PARAMS
from diffmah.main_sequence_sfr_eff import MEAN_SFR_MS_PARAMS
from diffmah.quenching_history import MEAN_Q_PARAMS
from measure_um_histories import measure_mean_histories
from load_um_histories import retrieve_centrals


DEFAULT_PARAM_DICT = OrderedDict()
DEFAULT_PARAM_DICT.update(MEAN_SFR_MS_PARAMS)
DEFAULT_PARAM_DICT.update(MEAN_Q_PARAMS)


def retrieve_all_um_targets(logm0arr, logm0_cens, log_ssfrh, smh, tsnap, tobs):
    tobs = np.linspace(1, 13.8, 25)
    _logm0arr = np.array((11.5, 11.75, 12, 12.25, 12.5, 13, 13.5, 14)).astype("f4")
    logm0arr = np.zeros_like(_logm0arr)

    all_targets = []
    for i, _logm0 in enumerate(_logm0arr):
        _x = measure_mean_histories(_logm0, logm0_cens, log_ssfrh, smh, tsnap, tobs)
        log_ssfrh_um, log_smh_um, logm0 = _x
        all_targets.append((log_ssfrh_um, log_smh_um))
        logm0arr[i] = logm0
    return logm0arr, all_targets


def _get_all_history_params(params):
    mean_mah_params = jax_np.array(list(MEAN_MAH_PARAMS.values())).astype("f4")
    n_sfr_params = len(MEAN_SFR_MS_PARAMS)
    mean_sfr_eff_params = params[0:n_sfr_params]
    mean_q_params = params[n_sfr_params:]
    return mean_mah_params, mean_sfr_eff_params, mean_q_params


def _mse_loss(prediction, target):
    diff = prediction - target
    return jax_np.sum(diff * diff) / diff.size


def mse_loss_global(params, global_loss_data):
    logt_table, indx_t0, dt, indx_pred = global_loss_data[0:4]
    logm0arr = global_loss_data[4]
    all_targets = global_loss_data[5]
    global_loss = 0.0
    for logm0, targets in zip(logm0arr, all_targets):
        mse_loss_logm0_data = (logt_table, indx_t0, dt, indx_pred, logm0, *targets)
        loss_logm0 = mse_loss_logm0(params, mse_loss_logm0_data)
        global_loss += loss_logm0
    n_logm0 = len(logm0arr)
    return global_loss / n_logm0


def mse_loss_logm0(params, data):
    _x = _get_all_history_params(params)
    mean_mah_params, mean_sfr_eff_params, mean_q_params = _x

    logt_table, indx_t0, dt, indx_pred, logm0 = data[0:5]
    log_sfrh_pred, log_smh_pred = _mean_log_mstar_history_jax_kern(
        mean_mah_params,
        mean_sfr_eff_params,
        mean_q_params,
        logm0,
        logt_table,
        indx_t0,
        dt,
        indx_pred,
    )
    log_ssfrh_pred = log_sfrh_pred - log_smh_pred
    log_ssfrh_target, log_smh_target = data[5:7]
    log_ssfr_loss = _mse_loss(log_ssfrh_pred, log_ssfrh_target)
    log_sm_loss = _mse_loss(log_smh_pred, log_smh_target)
    return log_ssfr_loss + log_sm_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n_opt_steps", type=int, help="Number of optimization steps",
    )
    parser.add_argument(
        "-outname", help="Name of the output file", default="best_fit_params.txt"
    )
    args = parser.parse_args()
    outname = args.outname
    n_opt_steps = args.n_opt_steps

    tobs = np.linspace(1, 13.8, 25)
    t_table = np.linspace(0.1, 13.85, 250)
    logt_table = np.log10(t_table)
    indx_t0 = t_table.size - 1
    dt = np.diff(t_table).mean()
    indx_pred = np.array([np.argmin(np.abs(t - t_table)) for t in tobs])

    cens, zarr_bpl, tarr_bpl, dtarr_bpl = retrieve_centrals()

    _logm0arr = np.array((11.5, 11.75, 12, 12.25, 12.5, 13, 13.5, 14)).astype("f4")
    logm0arr, all_targets = retrieve_all_um_targets(
        _logm0arr,
        cens["logm0"],
        cens["log_ssfrh"],
        cens["in_situ_mstar_history"],
        tarr_bpl,
        tobs,
    )

    global_mse_loss_data = (
        logt_table,
        indx_t0,
        dt,
        indx_pred,
        logm0arr,
        all_targets,
    )

    params_default = jax_np.array(list(DEFAULT_PARAM_DICT.values())).astype("f4")
    params_init = np.random.normal(loc=params_default, scale=0.05)
    loss_init = mse_loss_global(params_init, global_mse_loss_data)
    print("Initial global logloss = {0:.3f}".format(np.log10(loss_init)))
    np.savetxt("default_params.txt", params_default)

    opt_init, opt_update, get_params = jax_opt.adam(1e-3)
    opt_state = opt_init(params_init)

    start = time()
    for istep in range(n_opt_steps):
        loss, grads = value_and_grad(mse_loss_global, argnums=0)(
            get_params(opt_state), global_mse_loss_data
        )
        opt_state = opt_update(istep, grads, opt_state)
        if istep % 5 == 1:
            np.save("tmp_best_fit_params", get_params(opt_state))
            _msg = "...working on {0} of {1} logloss = {2:.4f}"
            print(_msg.format(istep, n_opt_steps, np.log10(loss)))

    print("\nFinal logloss = {0:.4f}".format(np.log10(loss)))
    best_fit_params = get_params(opt_state)
    np.savetxt(outname, best_fit_params)
    end = time()
    runtime = end - start
    runtime_per_eval = runtime / n_opt_steps

    msg = "\nTotal runtime = {0:.1f} seconds"
    print(msg.format(runtime))
    msg = "Runtime per likelihood evaluation = {0:.1f} seconds"
    print(msg.format(runtime_per_eval))

    # galaxy_history_fid_logm0_12 = get_mean_galaxy_history(12.0, tobs)
    # log_sfrh_target, log_smh_target = galaxy_history_fid_logm0_12
    # mse_loss_logm0_data = (
    #     logt_table,
    #     indx_t0,
    #     dt,
    #     indx_pred,
    #     12.0,
    #     log_sfrh_target,
    #     log_smh_target,
    # )
    #
    # default_params = jax_np.array(list(DEFAULT_PARAM_DICT.values())).astype("f4")
    #
    # rando_params = np.random.normal(loc=default_params, scale=0.1)
    # rando_loss_logm0_12 = mse_loss_logm0(rando_params, mse_loss_logm0_data)
    #
    # start = time()
    # loss, grads = value_and_grad(mse_loss_logm0, argnums=0)(
    #     rando_params, mse_loss_logm0_data
    # )
    # end = time()
    # grad_runtime = end - start
    # new_params = rando_params - 0.05 * grads
    # new_loss_logm0_12 = mse_loss_logm0(new_params, mse_loss_logm0_data)
    #
    #
    # all_targets = [get_mean_galaxy_history(lgm, tobs) for lgm in logm0arr]
    # global_mse_loss_data = (
    #     logt_table,
    #     indx_t0,
    #     dt,
    #     indx_pred,
    #     logm0arr,
    #     all_targets,
    # )
    # rando_global_loss = mse_loss_global(rando_params, global_mse_loss_data)
    # print("Rando global logloss = {0:.3f}".format(np.log10(rando_global_loss)))
    #
    # start = time()
    # loss, grads = value_and_grad(mse_loss_global, argnums=0)(
    #     rando_params, global_mse_loss_data
    # )
    # end = time()
    # global_grad_runtime = end - start
    # new_params = rando_params - 0.05 * grads
    # new_global_loss = mse_loss_global(new_params, global_mse_loss_data)
    # print("New global logloss = {0:.3f}".format(np.log10(new_global_loss)))
    # print("Runtime for global logloss = {0:.2f} seconds".format(global_grad_runtime))
