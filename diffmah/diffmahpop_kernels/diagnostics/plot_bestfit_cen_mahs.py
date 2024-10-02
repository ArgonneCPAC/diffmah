"""
"""

import os

import numpy as np
from jax import random as jran
from matplotlib import pyplot as plt

from .. import mc_diffmahpop_kernels_monocens as mcc

LGT0_SMDPL = np.log10(13.79)


def plot_singlepanel_mahs_centrals(cendata, diffmahpop_params, drn=""):
    ran_key = jran.key(0)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0.95, 14)
    ax.set_ylim(9.1, 15.5)

    msk_censample = cendata["t_obs"] > 13.5
    censample = cendata[msk_censample]

    # lgm_obs = 11.25
    ih = np.argmin(np.abs(censample["lgm_obs"] - 11.25))
    lgm_obs = censample["lgm_obs"][ih]
    t_obs = censample["t_obs"][ih]

    ylo = censample["mean_log_mah"][ih, :] - censample["std_log_mah"][ih, :]
    yhi = censample["mean_log_mah"][ih, :] + censample["std_log_mah"][ih, :]
    ax.fill_between(censample["t_table"][ih, :], ylo, yhi, color="lightgray", alpha=0.7)
    ax.plot(censample["t_table"][ih, :], censample["mean_log_mah"][ih, :], color="k")

    args = (
        diffmahpop_params,
        censample["t_table"][ih, :],
        lgm_obs,
        t_obs,
        ran_key,
        LGT0_SMDPL,
    )
    mean_log_mah, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(*args)
    ylo, yhi = mean_log_mah - std_log_mah, mean_log_mah + std_log_mah
    ax.plot(censample["t_table"][ih, :], mean_log_mah, "--", color="k")
    ax.plot(censample["t_table"][ih, :], ylo, ":", color="k")
    ax.plot(censample["t_table"][ih, :], yhi, ":", color="k")

    # lgm_obs = 12.75
    ih = np.argmin(np.abs(censample["lgm_obs"] - 12.75))
    lgm_obs = censample["lgm_obs"][ih]
    t_obs = censample["t_obs"][ih]

    ylo = censample["mean_log_mah"][ih, :] - censample["std_log_mah"][ih, :]
    yhi = censample["mean_log_mah"][ih, :] + censample["std_log_mah"][ih, :]
    ax.fill_between(censample["t_table"][ih, :], ylo, yhi, color="lightgray", alpha=0.7)
    ax.plot(censample["t_table"][ih, :], censample["mean_log_mah"][ih, :], color="k")

    args = (
        diffmahpop_params,
        censample["t_table"][ih, :],
        lgm_obs,
        t_obs,
        ran_key,
        LGT0_SMDPL,
    )
    mean_log_mah, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(*args)
    ylo, yhi = mean_log_mah - std_log_mah, mean_log_mah + std_log_mah
    ax.plot(censample["t_table"][ih, :], mean_log_mah, "--", color="k")
    ax.plot(censample["t_table"][ih, :], ylo, ":", color="k")
    ax.plot(censample["t_table"][ih, :], yhi, ":", color="k")

    # lgm_obs = 14.25
    ih = np.argmin(np.abs(censample["lgm_obs"] - 14.25))
    lgm_obs = censample["lgm_obs"][ih]
    t_obs = censample["t_obs"][ih]

    ylo = censample["mean_log_mah"][ih, :] - censample["std_log_mah"][ih, :]
    yhi = censample["mean_log_mah"][ih, :] + censample["std_log_mah"][ih, :]
    ax.fill_between(censample["t_table"][ih, :], ylo, yhi, color="lightgray", alpha=0.7)
    ax.plot(censample["t_table"][ih, :], censample["mean_log_mah"][ih, :], color="k")

    args = (
        diffmahpop_params,
        censample["t_table"][ih, :],
        lgm_obs,
        t_obs,
        ran_key,
        LGT0_SMDPL,
    )
    mean_log_mah, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(*args)
    ylo, yhi = mean_log_mah - std_log_mah, mean_log_mah + std_log_mah
    ax.plot(censample["t_table"][ih, :], mean_log_mah, "--", color="k")
    ax.plot(censample["t_table"][ih, :], ylo, ":", color="k")
    ax.plot(censample["t_table"][ih, :], yhi, ":", color="k")

    xlabel = ax.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
    ylabel = ax.set_ylabel(r"${\rm log_{10}M_{\rm h}/M_{\odot}}$")
    ax.set_title(r"${\rm host\ halos:\ }z=0$")

    os.makedirs(drn, exist_ok=True)
    outname = os.path.join(drn, "diffmahpop_mu_var_single_panel_cens_z0.png")
    fig.savefig(
        outname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
