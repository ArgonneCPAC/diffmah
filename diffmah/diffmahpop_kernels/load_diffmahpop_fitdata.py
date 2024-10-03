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
    cendata,
    satdata,
    t_obs_min_cen=2.5,
    t_obs_min_sat=2.5,
    lgm_obs_max_cen=14.5,
    lgm_obs_max_sat=13.5,
    drn=TASSO_DRN,
    n_sample_min=200,
):
    msk_cens = (cendata["t_obs"] > t_obs_min_cen) & (
        cendata["lgm_obs"] < lgm_obs_max_cen
    )
    msk_sats = (satdata["t_obs"] > t_obs_min_sat) & (
        satdata["lgm_obs"] < lgm_obs_max_sat
    )
    for key in cendata.keys():
        cendata[key] = cendata[key][msk_cens]
    for key in satdata.keys():
        satdata[key] = satdata[key][msk_sats]

    mah_samples_cens, mah_samples_sats = load_diffmahpop_target_samples(
        cendata, satdata, drn=drn
    )
    n_cens = len(cendata["lgm_obs"])
    n_sample_cens = np.array(
        [mah_samples_cens[ih][1].logm0.size for ih in range(n_cens)]
    )
    msk_n_sample_cens = n_sample_cens >= n_sample_min
    for key in cendata.keys():
        cendata[key] = cendata[key][msk_n_sample_cens]

    n_sats = len(satdata["lgm_obs"])
    n_sample_sats = np.array(
        [mah_samples_sats[ih][1].logm0.size for ih in range(n_sats)]
    )
    msk_n_sample_sats = n_sample_sats >= n_sample_min
    for key in satdata.keys():
        satdata[key] = satdata[key][msk_n_sample_sats]

    return cendata, satdata, mah_samples_cens, mah_samples_sats


def _mask_halo_samples(tdata, mah_samples, n_sample_min):
    n_halos = len(tdata["lgm_obs"])
    mah_samples_out = []
    for ih in range(n_halos):
        n_sample_arr = np.array(
            [mah_samples[ih][1].logm0.size for ih in range(n_halos)]
        )
        msk_n_sample = n_sample_arr >= n_sample_min

        mah_params_out_ih = [x[msk_n_sample] for x in mah_samples[ih][1]]
        mah_params_out_ih = DEFAULT_MAH_PARAMS._make(mah_params_out_ih)

        log_mah_out_ih = mah_samples[ih][2][msk_n_sample]
        log_mah_rescaled_out_ih = mah_samples[ih][3][msk_n_sample]

        t_table_sample = mah_samples[ih][0]
        line_out_ih = (
            t_table_sample,
            mah_params_out_ih,
            log_mah_out_ih,
            log_mah_rescaled_out_ih,
        )

        mah_samples_out.append(line_out_ih)

    return tdata_out, mah_samples_out


def load_diffmahpop_target_mu_var(drn=TASSO_DRN):
    cendata, satdata = _load_diffmahpop_mu_var_targets(drn=drn)
    cendata, satdata = get_target_subset_for_fitting(cendata, satdata)
    return cendata, satdata


def load_diffmahpop_target_samples(cendata, satdata, drn=TASSO_DRN):
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
