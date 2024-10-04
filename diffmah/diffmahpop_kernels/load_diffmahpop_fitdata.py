"""
"""

import os

import h5py
import numpy as np

from ..diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop

try:
    from astropy.table import Table
except ImportError:
    pass

TASSO_DRN = "/Users/aphearin/work/DATA/diffmahpop_data/NM_12_NT_9_ISTART_0_IEND_576"
N_SAMPLE_MIN = 200

PAT_CENS = "lgm_{0:.2f}_t_{1:.2f}_mah_params.cens.npy"
PAT_SATS = "lgm_{0:.2f}_t_{1:.2f}_mah_params.sats.npy"


def load_diffmahpop_targets(
    drn=TASSO_DRN,
    t_obs_min_cen=2.5,
    t_obs_min_sat=2.5,
    lgm_obs_max_cen=14.5,
    lgm_obs_max_sat=13.5,
    n_sample_min=N_SAMPLE_MIN,
):
    """Load target mean and variance along with subset of underlying MAH samples"""
    cendata, satdata = _load_diffmahpop_mu_var_targets(drn=drn)
    cendata, satdata = _add_n_sample_column(cendata, satdata)
    cendata, satdata = _attach_samples(
        drn,
        cendata,
        satdata,
        t_obs_min_cen,
        t_obs_min_sat,
        lgm_obs_max_cen,
        lgm_obs_max_sat,
        n_sample_min=n_sample_min,
    )
    return cendata, satdata


def rescale_target_log_mahs(log_mah_table, lgm_obs):
    delta_lgm_obs_table = log_mah_table[:, -1] - lgm_obs
    log_mah_sample_target = log_mah_table - delta_lgm_obs_table.reshape((-1, 1))
    return log_mah_sample_target


def _load_flat_hdf5(fn):
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    return data


def _load_diffmahpop_mu_var_targets(drn):
    cendata = Table(_load_flat_hdf5(os.path.join(drn, "mean_var_cens.h5")))
    satdata = Table(_load_flat_hdf5(os.path.join(drn, "mean_var_sats.h5")))
    cendata["mean_log_mah_rescaled"] = rescale_target_log_mahs(
        cendata["mean_log_mah"], cendata["lgm_obs"]
    )
    satdata["mean_log_mah_rescaled"] = rescale_target_log_mahs(
        satdata["mean_log_mah"], satdata["lgm_obs"]
    )
    return cendata, satdata


def _add_n_sample_column(cendata, satdata):

    n_cens = len(cendata["lgm_obs"])
    n_sats = len(satdata["lgm_obs"])

    cendata["n_samples"] = 0
    satdata["n_samples"] = 0

    for ih in range(n_cens):
        fn = os.path.join(
            TASSO_DRN, PAT_CENS.format(cendata["lgm_obs"][ih], cendata["t_obs"][ih])
        )
        all_samples_cens = np.load(fn)
        cendata["n_samples"][ih] = all_samples_cens.shape[0]

    for ih in range(n_sats):
        fn = os.path.join(
            TASSO_DRN, PAT_SATS.format(satdata["lgm_obs"][ih], satdata["t_obs"][ih])
        )
        all_samples_sats = np.load(fn)
        satdata["n_samples"][ih] = all_samples_sats.shape[0]

    return cendata, satdata


def _attach_samples(
    drn,
    cendata,
    satdata,
    t_obs_min_cen,
    t_obs_min_sat,
    lgm_obs_max_cen,
    lgm_obs_max_sat,
    n_sample_min,
):
    msk_n_sample_cens = cendata["n_samples"] >= n_sample_min
    msk_t_obs_cens = cendata["t_obs"] > t_obs_min_cen
    msk_lgm_obs_cens = cendata["lgm_obs"] < lgm_obs_max_cen
    msk_cens = msk_t_obs_cens & msk_lgm_obs_cens & msk_n_sample_cens

    msk_n_sample_sats = satdata["n_samples"] >= n_sample_min
    msk_t_obs_sats = satdata["t_obs"] > t_obs_min_sat
    msk_lgm_obs_sats = satdata["lgm_obs"] < lgm_obs_max_sat
    msk_sats = msk_t_obs_sats & msk_lgm_obs_sats & msk_n_sample_sats

    cendata = cendata[msk_cens]
    satdata = satdata[msk_sats]

    n_cendata = len(cendata)
    n_satdata = len(satdata)

    cendata["logm0_samples"] = np.zeros((n_cendata, n_sample_min)) - 1
    cendata["logtc_samples"] = np.zeros((n_cendata, n_sample_min)) - 1
    cendata["early_index_samples"] = np.zeros((n_cendata, n_sample_min)) - 1
    cendata["late_index_samples"] = np.zeros((n_cendata, n_sample_min)) - 1
    cendata["t_peak_samples"] = np.zeros((n_cendata, n_sample_min)) - 1

    satdata["logm0_samples"] = np.zeros((n_satdata, n_sample_min)) - 1
    satdata["logtc_samples"] = np.zeros((n_satdata, n_sample_min)) - 1
    satdata["early_index_samples"] = np.zeros((n_satdata, n_sample_min)) - 1
    satdata["late_index_samples"] = np.zeros((n_satdata, n_sample_min)) - 1
    satdata["t_peak_samples"] = np.zeros((n_satdata, n_sample_min)) - 1

    for ih in range(len(cendata)):
        fn_cens = os.path.join(
            drn, PAT_CENS.format(cendata["lgm_obs"][ih], cendata["t_obs"][ih])
        )
        all_cens_ih = np.load(fn_cens)
        n_all_cens = all_cens_ih.shape[0]
        assert n_all_cens >= n_sample_min
        indx_cens = np.random.choice(
            np.arange(n_all_cens).astype(int), n_sample_min, replace=False
        )
        cendata["logm0_samples"][ih, :] = all_cens_ih[indx_cens, 0]
        cendata["logtc_samples"][ih, :] = all_cens_ih[indx_cens, 1]
        cendata["early_index_samples"][ih, :] = all_cens_ih[indx_cens, 2]
        cendata["late_index_samples"][ih, :] = all_cens_ih[indx_cens, 3]
        cendata["t_peak_samples"][ih, :] = all_cens_ih[indx_cens, 4]

    for ih in range(len(satdata)):
        fn_sats = os.path.join(
            drn, PAT_SATS.format(satdata["lgm_obs"][ih], satdata["t_obs"][ih])
        )
        all_sats_ih = np.load(fn_sats)
        n_all_sats = all_sats_ih.shape[0]
        assert n_all_sats >= n_sample_min
        indx_sats = np.random.choice(
            np.arange(n_all_sats).astype(int), n_sample_min, replace=False
        )
        satdata["logm0_samples"][ih, :] = all_sats_ih[indx_sats, 0]
        satdata["logtc_samples"][ih, :] = all_sats_ih[indx_sats, 1]
        satdata["early_index_samples"][ih, :] = all_sats_ih[indx_sats, 2]
        satdata["late_index_samples"][ih, :] = all_sats_ih[indx_sats, 3]
        satdata["t_peak_samples"][ih, :] = all_sats_ih[indx_sats, 4]

    mah_keys = (
        "logm0_samples",
        "logtc_samples",
        "early_index_samples",
        "late_index_samples",
    )

    n_times = cendata["t_table"].shape[1]
    cendata["log_mah_samples"] = np.zeros((n_cendata, n_sample_min, n_times))
    cendata["log_mah_rescaled_samples"] = np.zeros((n_cendata, n_sample_min, n_times))
    for ih in range(len(cendata)):
        mah_params_cens = DEFAULT_MAH_PARAMS._make(
            [cendata[key][ih, :] for key in mah_keys]
        )
        t_table_sample = cendata["t_table"][ih, :]
        t_peak_cens = cendata["t_peak_samples"][ih]
        __, log_mah_cens = mah_halopop(
            mah_params_cens, t_table_sample, t_peak_cens, np.log10(13.79)
        )
        log_mah_cens_rescaled = rescale_target_log_mahs(
            log_mah_cens, cendata["lgm_obs"][ih]
        )
        cendata["log_mah_samples"][ih, :] = log_mah_cens
        cendata["log_mah_rescaled_samples"][ih, :] = log_mah_cens_rescaled

    satdata["log_mah_samples"] = np.zeros((n_satdata, n_sample_min, n_times))
    satdata["log_mah_rescaled_samples"] = np.zeros((n_satdata, n_sample_min, n_times))
    for ih in range(len(satdata)):
        mah_params_sats = DEFAULT_MAH_PARAMS._make(
            [satdata[key][ih, :] for key in mah_keys]
        )
        t_table_sample = satdata["t_table"][ih, :]
        t_peak_sats = satdata["t_peak_samples"][ih]
        __, log_mah_sats = mah_halopop(
            mah_params_sats, t_table_sample, t_peak_sats, np.log10(13.79)
        )
        log_mah_sats_rescaled = rescale_target_log_mahs(
            log_mah_sats, satdata["lgm_obs"][ih]
        )
        satdata["log_mah_samples"][ih, :] = log_mah_sats
        satdata["log_mah_rescaled_samples"][ih, :] = log_mah_sats_rescaled

    return cendata, satdata
