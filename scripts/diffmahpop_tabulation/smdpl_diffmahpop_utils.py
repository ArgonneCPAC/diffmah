"""This module stores helper functions for tabulating target data for DiffmahPop.
The primary function is smdpl_diffmahpop_subvolume_loop, which loops over subvolumes
of diffmah fits to SMDPL and stores halo samples binned by (lgm_obs, t_obs).
"""

NH_CUT = 100

import os
import subprocess
from glob import glob

import h5py
import numpy as np
from jax import jit as jjit
from umachine_pyio.load_mock import load_mock_from_binaries

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop

TASSO_ROOT_DRN = "/Users/aphearin/work/DATA/SMDPL/dr1_no_merging_upidh/"
TASSO_SFHCAT_DRN = os.path.join(TASSO_ROOT_DRN, "sfh_binary_catalogs/a_1.000000")
LCRC_SFHCAT_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000"
)

LCRC_DIFFMAH_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/diffmah_tpeak_fits"
)
TASSO_DIFFMAH_DRN = "/Users/aphearin/work/DATA/diffstar_data/SMDPL"
N_SUBVOL_SMDPL = 576
LGMH_MIN, LGMH_MAX = 11, 14.75
N_LGM_BINS = 12

T0_SMDPL = 13.7976158

T_TABLE_MIN = 0.5
T_OBS_TARGET_MIN = 2.0
N_TIMES = 9

TASSO_OUTDRN = ""
N_HALOS_MAX = 20_000
N_HALOS_PER_SUBVOL = N_HALOS_MAX // N_SUBVOL_SMDPL
FMT_OUT = "%.4f"
HEADER_OUT = " ".join(DEFAULT_MAH_PARAMS._fields) + " t_peak"

NUM_DIFFMAH_PARAMS = len(DEFAULT_MAH_PARAMS) + 1  # t_peak currently not a param


@jjit
def _get_lgm_thresh_from_t_obs(t):
    lgm_thresh = 15 + 0.075 * (t - 15.0)
    return lgm_thresh


def lgmarr_table_from_t_obs_kern(t, n_m=N_LGM_BINS):
    lgm_thresh = _get_lgm_thresh_from_t_obs(t)
    lgmarr = np.linspace(LGMH_MIN, lgm_thresh, n_m)
    return lgmarr


def _get_subvol_bnpat(subvol, lgm_obs, t_obs, censat):
    bnpat = f"subvol_{subvol}_lgm_{lgm_obs:.2f}_t_{t_obs:.2f}_mah_params.{censat}.txt"
    return bnpat


def _get_output_drn(drn, n_lgm, n_t, istart, iend):
    drnpat = f"NM_{n_lgm}_NT_{n_t}_ISTART_{istart}_IEND_{iend}"
    return os.path.join(drn, drnpat)


def get_t_obs_table(t0, n_times):
    t_table = np.linspace(T_OBS_TARGET_MIN, t0, n_times)
    return t_table


def get_t_table_for_t_obs(t_obs, n_times):
    t_table = np.linspace(T_TABLE_MIN, t_obs, n_times)
    return t_table


def smdpl_diffmahpop_subvolume_loop(
    outdrn,
    diffmah_drn=LCRC_DIFFMAH_DRN,
    sfhcat_drn=LCRC_SFHCAT_DRN,
    istart=0,
    iend=N_SUBVOL_SMDPL,
    n_m=N_LGM_BINS,
    n_t=N_TIMES,
):
    for subvol in range(istart, iend):
        for target_data in generate_diffmahpop_targets_for_subvol(
            subvol, diffmah_drn=diffmah_drn, sfhcat_drn=sfhcat_drn, n_m=n_m, n_t=n_t
        ):
            lgm_obs, t_obs, mah_params_sample, t_peak_sample, upid_sample = target_data
            bnout_cens = _get_subvol_bnpat(subvol, lgm_obs, t_obs, "cens")
            bnout_sats = _get_subvol_bnpat(subvol, lgm_obs, t_obs, "sats")
            fnout_cens = os.path.join(outdrn, bnout_cens)
            fnout_sats = os.path.join(outdrn, bnout_sats)

            msk_cens = upid_sample == -1
            n_halos = t_peak_sample.size
            n_cens = msk_cens.sum()
            n_sats = n_halos - n_cens
            msk_sats = ~msk_cens

            # Output data for centrals
            if n_cens > 0:
                pcens = [x[msk_cens][:N_HALOS_PER_SUBVOL] for x in mah_params_sample]
                tpcens = t_peak_sample[msk_cens][:N_HALOS_PER_SUBVOL]
                data_cens = np.array([*pcens, tpcens]).T
                np.savetxt(fnout_cens, data_cens, fmt=FMT_OUT, header=HEADER_OUT)

            # Output data for satellites
            if n_sats > 0:
                psats = [x[msk_sats][:N_HALOS_PER_SUBVOL] for x in mah_params_sample]
                tpsats = t_peak_sample[msk_sats][:N_HALOS_PER_SUBVOL]
                data_sats = np.array([*psats, tpsats]).T
                np.savetxt(fnout_sats, data_sats, fmt=FMT_OUT, header=HEADER_OUT)

    collate_subvolume_samples(outdrn, T0_SMDPL, n_m, n_t, istart, iend)


def generate_diffmahpop_targets_for_subvol(
    subvol,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=LCRC_DIFFMAH_DRN,
    sfhcat_drn=LCRC_SFHCAT_DRN,
    n_m=N_LGM_BINS,
    n_t=N_TIMES,
):
    diffmah_data, mah_params, t_peak, upidh = load_diffmah_subvolume(
        subvol,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        sfhcat_drn=sfhcat_drn,
    )
    t_table_smdpl = np.loadtxt(os.path.join(sfhcat_drn, "smdpl_cosmic_time.txt"))
    t0 = t_table_smdpl[-1]
    t_obs_arr = get_t_obs_table(t0, n_t)

    for t_obs in t_obs_arr:
        indx_t = np.argmin(np.abs(t_obs - t_table_smdpl))
        upid = upidh[:, indx_t]
        t_table, log_mah_table = get_log_mah_tables_for_t_obs(
            mah_params, t_peak, t_obs, t0
        )
        lgm_obs_arr = lgmarr_table_from_t_obs_kern(t_obs, n_m=n_m)
        for lgm_obs in lgm_obs_arr:
            mah_params_sample, t_peak_sample, upid_sample = get_target_mah_params(
                log_mah_table, lgm_obs, mah_params, t_peak, upid
            )
            n_targets = t_peak_sample.shape[0]
            if n_targets > 0:
                yield lgm_obs, t_obs, mah_params_sample, t_peak_sample, upid_sample


def _load_flat_hdf5(fn):
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    return data


def load_diffmah_subvolume(
    subvol,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=LCRC_DIFFMAH_DRN,
    sfhcat_drn=LCRC_SFHCAT_DRN,
):
    nchar_subvol = len(str(n_subvol_tot))
    diffmah_bnpat = "subvol_{}_diffmah_fits.h5"
    subvol_str = f"{subvol:0{nchar_subvol}d}"
    diffmah_bn = diffmah_bnpat.format(subvol_str)
    diffmah_fn = os.path.join(diffmah_drn, diffmah_bn)
    diffmah_data = _load_flat_hdf5(diffmah_fn)

    subvols = [subvol]
    galprops = ["halo_id", "upid_history"]
    halocat_data = load_mock_from_binaries(subvols, sfhcat_drn, galprops)

    # throw out halos with a bad fit
    msk = diffmah_data["loss"] != -1
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_data[key][msk] for key in DEFAULT_MAH_PARAMS._fields]
    )
    t_peak = diffmah_data["t_peak"][msk]

    upidh = halocat_data["upid_history"][msk]

    return diffmah_data, mah_params, t_peak, upidh


def get_log_mah_tables_for_t_obs(mah_params, t_peak, t_obs, t0):
    t_table = get_t_table_for_t_obs(t_obs, 2)
    lgt0 = np.log10(t0)
    log_mah_table = mah_halopop(mah_params, t_table, t_peak, lgt0)[1]
    return t_table, log_mah_table


def get_target_mah_params(
    log_mah_table, lgm_obs, mah_params, t_peak, upid, lgm_obs_bin_width=0.15
):
    delta_lgm_obs_table = log_mah_table[:, -1] - lgm_obs
    msk_sample = np.abs(delta_lgm_obs_table) < lgm_obs_bin_width
    mah_params_sample = mah_params._make([x[msk_sample] for x in mah_params])
    t_peak_sample = t_peak[msk_sample]
    upid_sample = upid[msk_sample]
    return mah_params_sample, t_peak_sample, upid_sample


def rescale_target_log_mahs(log_mah_table, lgm_obs):
    delta_lgm_obs_table = log_mah_table[:, -1] - lgm_obs
    log_mah_sample_target = log_mah_table - delta_lgm_obs_table.reshape((-1, 1))
    return log_mah_sample_target


def collate_subvolume_samples(drn, t0, n_m, n_t, istart, iend):
    t_obs_arr = get_t_obs_table(t0, n_t)

    drnout = _get_output_drn(drn, n_m, n_t, istart, iend)
    os.makedirs(drnout)

    for t_obs in t_obs_arr:
        lgm_obs_arr = lgmarr_table_from_t_obs_kern(t_obs, n_m=n_m)

        for lgm_obs in lgm_obs_arr:

            # centrals
            bnpat_cens = _get_subvol_bnpat("*", lgm_obs, t_obs, "cens")
            fnlist_cens = glob(os.path.join(drn, bnpat_cens))
            bnlist_cens = [os.path.basename(fn) for fn in fnlist_cens]

            cendata_collector = []
            for fn in fnlist_cens:
                X = np.loadtxt(fn).reshape((-1, NUM_DIFFMAH_PARAMS))
                cendata_collector.append(X)

            if len(cendata_collector) > 0:
                cendata = np.concatenate(cendata_collector)
                cendata = cendata.reshape((-1, NUM_DIFFMAH_PARAMS))
                bnout = "_".join(bnlist_cens[0].split("_")[2:])
                bnout = bnout.replace(".txt", "")
                fnout = os.path.join(drnout, bnout)
                np.save(fnout, cendata)

            # satellites
            bnpat_sats = _get_subvol_bnpat("*", lgm_obs, t_obs, "sats")
            fnlist_sats = glob(os.path.join(drn, bnpat_sats))
            bnlist_sats = [os.path.basename(fn) for fn in fnlist_sats]

            satdata_collector = []
            for fn in fnlist_sats:
                X = np.loadtxt(fn).reshape((-1, NUM_DIFFMAH_PARAMS))
                satdata_collector.append(X)

            if len(satdata_collector) > 0:
                satdata = np.concatenate(satdata_collector)
                satdata = satdata.reshape((-1, NUM_DIFFMAH_PARAMS))
                bnout = "_".join(bnlist_sats[0].split("_")[2:])
                bnout = bnout.replace(".txt", "")
                fnout = os.path.join(drnout, bnout)
                np.save(fnout, satdata)

    for subvol in range(10):
        fname_pat = os.path.join(drn, f"subvol_{subvol}*")
        fname_list = glob(fname_pat)
        if len(fname_list) > 0:
            command = "rm " + fname_pat
            subprocess.check_output(command, shell=True)


def tabulate_target_means_vars(
    mah_params_drn, nh_min=NH_CUT, n_t_target=25, rescale=True
):
    """"""
    lgt0 = np.log10(T0_SMDPL)

    cens_fname_list = glob(os.path.join(mah_params_drn, "*cens.npy"))
    cens_bname_list = [os.path.basename(fn) for fn in cens_fname_list]

    sats_fname_list = glob(os.path.join(mah_params_drn, "*sats.npy"))
    sats_bname_list = [os.path.basename(fn) for fn in sats_fname_list]

    cens_lgm_obs_arr = np.array([float(bn.split("_")[1]) for bn in cens_bname_list])
    cens_t_obs_arr = np.array([float(bn.split("_")[3]) for bn in cens_bname_list])

    sats_lgm_obs_arr = np.array([float(bn.split("_")[1]) for bn in sats_bname_list])
    sats_t_obs_arr = np.array([float(bn.split("_")[3]) for bn in sats_bname_list])

    cens_target_collector = []
    cens_gen = zip(cens_fname_list, cens_lgm_obs_arr, cens_t_obs_arr)
    for fn, lgm_obs, t_obs in cens_gen:
        diffmah_data = np.load(fn).reshape((-1, NUM_DIFFMAH_PARAMS))
        nh = diffmah_data.shape[0]
        t_table = get_t_table_for_t_obs(t_obs, n_t_target)
        if nh > nh_min:
            t_peak = diffmah_data[:, 4]
            mah_params = DEFAULT_MAH_PARAMS._make(
                [diffmah_data[:, i] for i in range(NUM_DIFFMAH_PARAMS - 1)]
            )

            log_mah_table = mah_halopop(mah_params, t_table, t_peak, lgt0)[1]
            if rescale:
                log_mah_table = rescale_target_log_mahs(log_mah_table, lgm_obs)
            mean_log_mah = np.mean(log_mah_table, axis=0)
            std_log_mah = np.std(log_mah_table, axis=0)
            frac_peaked = np.array([np.mean(t_peak < t) for t in t_table])
            target = (lgm_obs, t_obs, mean_log_mah, std_log_mah, frac_peaked, t_table)
            cens_target_collector.append(target)

    cen_targets = dict()
    cen_targets["lgm_obs"] = np.array([x[0] for x in cens_target_collector])
    cen_targets["t_obs"] = np.array([x[1] for x in cens_target_collector])
    cen_targets["mean_log_mah"] = np.array([x[2] for x in cens_target_collector])
    cen_targets["std_log_mah"] = np.array([x[3] for x in cens_target_collector])
    cen_targets["frac_peaked"] = np.array([x[4] for x in cens_target_collector])
    cen_targets["t_table"] = np.array([x[5] for x in cens_target_collector])

    sats_target_collector = []
    sats_gen = zip(sats_fname_list, sats_lgm_obs_arr, sats_t_obs_arr)
    for fn, lgm_obs, t_obs in sats_gen:
        diffmah_data = np.load(fn).reshape((-1, NUM_DIFFMAH_PARAMS))
        nh = diffmah_data.shape[0]
        t_table = get_t_table_for_t_obs(t_obs, n_t_target)
        if nh > nh_min:
            t_peak = diffmah_data[:, 4]
            mah_params = DEFAULT_MAH_PARAMS._make(
                [diffmah_data[:, i] for i in range(NUM_DIFFMAH_PARAMS - 1)]
            )

            log_mah_table = mah_halopop(mah_params, t_table, t_peak, lgt0)[1]
            if rescale:
                log_mah_table = rescale_target_log_mahs(log_mah_table, lgm_obs)
            mean_log_mah = np.mean(log_mah_table, axis=0)
            std_log_mah = np.std(log_mah_table, axis=0)
            frac_peaked = np.array([np.mean(t_peak < t) for t in t_table])
            target = (lgm_obs, t_obs, mean_log_mah, std_log_mah, frac_peaked, t_table)
            sats_target_collector.append(target)

    sat_targets = dict()
    sat_targets["lgm_obs"] = np.array([x[0] for x in sats_target_collector])
    sat_targets["t_obs"] = np.array([x[1] for x in sats_target_collector])
    sat_targets["mean_log_mah"] = np.array([x[2] for x in sats_target_collector])
    sat_targets["std_log_mah"] = np.array([x[3] for x in sats_target_collector])
    sat_targets["frac_peaked"] = np.array([x[4] for x in sats_target_collector])
    sat_targets["t_table"] = np.array([x[5] for x in sats_target_collector])

    return cen_targets, sat_targets


def write_mean_variance_to_disk(mah_params_drn, fnout_cens, fnout_sats):
    """This function creates the mean and variance target data used to fit diffmahpop.
    For example, this function created NM_12_NT_9_ISTART_0_IEND_576_mu_var_cens.h5
    """
    cen_targets, sat_targets = tabulate_target_means_vars(mah_params_drn)

    with h5py.File(fnout_cens, "w") as hdfout:
        for key in cen_targets.keys():
            hdfout[key] = cen_targets[key]

    with h5py.File(fnout_sats, "w") as hdfout:
        for key in sat_targets.keys():
            hdfout[key] = sat_targets[key]
