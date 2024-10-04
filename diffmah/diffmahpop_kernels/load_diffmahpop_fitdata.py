"""
"""

import os

import h5py
import numpy as np

from ..diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop

TASSO_DRN = "/Users/aphearin/work/DATA/diffmahpop_data/NM_12_NT_9_ISTART_0_IEND_576"
N_SAMPLE_MIN = 200


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
    cendata, satdata, mah_samples_cens, mah_samples_sats = (
        get_target_subset_for_fitting(
            drn,
            cendata,
            satdata,
            t_obs_min_cen,
            t_obs_min_sat,
            lgm_obs_max_cen,
            lgm_obs_max_sat,
            n_sample_min,
        )
    )
    X = np.array([mah_samples_cens[ih][0] for ih in range(len(mah_samples_cens))])
    assert np.allclose(X, cendata["t_table"])
    cendata["log_mah_samples"] = np.array(
        [mah_samples_cens[ih][2] for ih in range(len(mah_samples_cens))]
    )
    cendata["log_mah_rescaled_samples"] = np.array(
        [mah_samples_cens[ih][3] for ih in range(len(mah_samples_cens))]
    )
    Y = np.array([mah_samples_sats[ih][0] for ih in range(len(mah_samples_sats))])
    assert np.allclose(Y, satdata["t_table"])
    satdata["log_mah_samples"] = np.array(
        [mah_samples_sats[ih][2] for ih in range(len(mah_samples_sats))]
    )
    satdata["log_mah_rescaled_samples"] = np.array(
        [mah_samples_sats[ih][3] for ih in range(len(mah_samples_sats))]
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
    cendata = _load_flat_hdf5(os.path.join(drn, "mean_var_cens.h5"))
    satdata = _load_flat_hdf5(os.path.join(drn, "mean_var_sats.h5"))
    cendata["mean_log_mah_rescaled"] = rescale_target_log_mahs(
        cendata["mean_log_mah"], cendata["lgm_obs"]
    )
    satdata["mean_log_mah_rescaled"] = rescale_target_log_mahs(
        satdata["mean_log_mah"], satdata["lgm_obs"]
    )
    return cendata, satdata


def get_target_subset_for_fitting(
    drn,
    cendata,
    satdata,
    t_obs_min_cen,
    t_obs_min_sat,
    lgm_obs_max_cen,
    lgm_obs_max_sat,
    n_sample_min,
):
    mah_samples_cens, mah_samples_sats = load_diffmahpop_target_samples(
        drn, cendata, satdata
    )

    n_cens = len(cendata["lgm_obs"])
    n_sample_cens = np.array(
        [mah_samples_cens[ih][1].logm0.size for ih in range(n_cens)]
    )
    msk_n_sample_cens = n_sample_cens >= n_sample_min
    msk_t_obs_cens = cendata["t_obs"] > t_obs_min_cen
    msk_lgm_obs_cens = cendata["lgm_obs"] < lgm_obs_max_cen
    msk_cens = msk_t_obs_cens & msk_lgm_obs_cens & msk_n_sample_cens

    n_sats = len(satdata["lgm_obs"])
    n_sample_sats = np.array(
        [mah_samples_sats[ih][1].logm0.size for ih in range(n_sats)]
    )
    msk_n_sample_sats = n_sample_sats >= n_sample_min
    msk_t_obs_sats = satdata["t_obs"] > t_obs_min_sat
    msk_lgm_obs_sats = satdata["lgm_obs"] < lgm_obs_max_sat
    msk_sats = msk_t_obs_sats & msk_lgm_obs_sats & msk_n_sample_sats

    for key in cendata.keys():
        cendata[key] = cendata[key][msk_cens]

    for key in satdata.keys():
        satdata[key] = satdata[key][msk_sats]

    ih_keep_cens = np.arange(n_cens)[msk_cens].astype(int)
    ih_keep_sats = np.arange(n_sats)[msk_sats].astype(int)

    mah_samples_cens = _mask_halo_samples(mah_samples_cens, ih_keep_cens, n_sample_min)
    mah_samples_sats = _mask_halo_samples(mah_samples_sats, ih_keep_sats, n_sample_min)

    return cendata, satdata, mah_samples_cens, mah_samples_sats


def _mask_halo_samples(mah_samples, ih_keep_arr, n_sample_min):
    mah_samples_out = []
    for ih in ih_keep_arr:

        mah_params_out_ih = [x[:n_sample_min] for x in mah_samples[ih][1]]
        mah_params_out_ih = DEFAULT_MAH_PARAMS._make(mah_params_out_ih)

        log_mah_out_ih = mah_samples[ih][2][:n_sample_min]
        log_mah_rescaled_out_ih = mah_samples[ih][3][:n_sample_min]

        t_table_sample = mah_samples[ih][0]
        line_out_ih = (
            t_table_sample,
            mah_params_out_ih,
            log_mah_out_ih,
            log_mah_rescaled_out_ih,
        )

        mah_samples_out.append(line_out_ih)

    return mah_samples_out


def load_diffmahpop_target_samples(drn, cendata, satdata):
    pat = "lgm_{0:.2f}_t_{1:.2f}_mah_params.{2}.npy"

    # centrals
    mah_samples_cens = []
    n_cens = len(cendata["lgm_obs"])
    for ih in range(n_cens):
        bn = pat.format(cendata["lgm_obs"][ih], cendata["t_obs"][ih], "sats")
        fn = os.path.join(drn, bn)
        cen_sample = np.load(fn)

        t_table_sample = cendata["t_table"][ih, :]
        mah_params_cens = DEFAULT_MAH_PARAMS._make([cen_sample[:, i] for i in range(4)])
        __, log_mah_cens = mah_halopop(
            mah_params_cens, t_table_sample, cen_sample[:, 4], np.log10(13.79)
        )
        log_mah_cens_rescaled = rescale_target_log_mahs(
            log_mah_cens, cendata["lgm_obs"][ih]
        )
        mah_samples_cens.append(
            (t_table_sample, mah_params_cens, log_mah_cens, log_mah_cens_rescaled)
        )

    # satellites
    mah_samples_sats = []
    n_sats = len(satdata["lgm_obs"])
    for ih in range(n_sats):
        bn = pat.format(satdata["lgm_obs"][ih], satdata["t_obs"][ih], "sats")
        fn = os.path.join(drn, bn)
        sat_sample = np.load(fn)

        t_table_sample = satdata["t_table"][ih, :]
        mah_params_sats = DEFAULT_MAH_PARAMS._make([sat_sample[:, i] for i in range(4)])
        __, log_mah_sats = mah_halopop(
            mah_params_sats, t_table_sample, sat_sample[:, 4], np.log10(13.79)
        )
        log_mah_sats_rescaled = rescale_target_log_mahs(
            log_mah_sats, satdata["lgm_obs"][ih]
        )
        mah_samples_sats.append(
            (t_table_sample, mah_params_sats, log_mah_sats, log_mah_sats_rescaled)
        )

    return mah_samples_cens, mah_samples_sats
