"""
"""

import os

import numpy as np
from jax import random as jran

from .. import mc_bimod_cens as mcc
from .defaults import mblue, mred

try:
    from matplotlib import cm
    from matplotlib import pyplot as plt
except ImportError:
    pass

LGT0_SMDPL = np.log10(13.79)

__all__ = (
    "plot_singlepanel_mahs_centrals_z0",
    "plot_singlepanel_mahs_centrals_z2",
    "plot_cen_mah_4panel_residuals",
)


def plot_singlepanel_mahs_centrals_z0(cendata, diffmahpop_params, drn="FIGS"):
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


def plot_singlepanel_mahs_centrals_z2(cendata, diffmahpop_params, drn="FIGS"):
    ran_key = jran.key(0)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0.95, 3.5)
    ax.set_ylim(9.1, 15.5)

    msk_censample = np.abs(cendata["t_obs"] - 3) < 0.5
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
    ax.set_title(r"${\rm host\ halos:\ }z=2$")

    os.makedirs(drn, exist_ok=True)
    outname = os.path.join(drn, "diffmahpop_mu_var_single_panel_cens_z2.png")
    fig.savefig(
        outname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )


def plot_cen_mah_4panel_residuals(cendata, diffmahpop_params, drn="FIGS"):
    ran_key = jran.key(0)
    n_mass = 15
    lgm_obs_arr = np.linspace(11, 14.5, n_mass)
    mass_colors = cm.coolwarm(np.linspace(0, 1, n_mass))  # blue first

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    ax0.set_ylim(-0.45, 0.45)
    t_obs_panel_list = [4.0, 7.0, 10.0, 13.0]

    ##############################################
    t_obs_panel = t_obs_panel_list[0]
    ax = ax0
    ax.set_title(f"host halos: t_obs = {t_obs_panel:.1f}")

    for ic, lgm_obs_plot in enumerate(lgm_obs_arr):
        ih = np.argmin(
            np.abs(cendata["t_obs"] - t_obs_panel)
            + np.abs(cendata["lgm_obs"] - lgm_obs_plot)
        )
        tarr = cendata["t_table"][ih]
        mean_log_mah_target = cendata["mean_log_mah"][ih, :]

        lgm_obs = cendata["lgm_obs"][ih]
        t_obs = cendata["t_obs"][ih]
        args = (diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, LGT0_SMDPL)
        mean_log_mah_pred, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(
            *args
        )
        ax.plot(tarr, mean_log_mah_pred - mean_log_mah_target, color=mass_colors[ic])

    ##############################################
    t_obs_panel = t_obs_panel_list[1]
    ax = ax1
    ax.set_title(f"host halos: t_obs = {t_obs_panel:.1f}")

    for ic, lgm_obs_plot in enumerate(lgm_obs_arr):
        ih = np.argmin(
            np.abs(cendata["t_obs"] - t_obs_panel)
            + np.abs(cendata["lgm_obs"] - lgm_obs_plot)
        )
        tarr = cendata["t_table"][ih]
        mean_log_mah_target = cendata["mean_log_mah"][ih, :]

        lgm_obs = cendata["lgm_obs"][ih]
        t_obs = cendata["t_obs"][ih]
        args = (diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, LGT0_SMDPL)
        mean_log_mah_pred, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(
            *args
        )
        ax.plot(tarr, mean_log_mah_pred - mean_log_mah_target, color=mass_colors[ic])

    ##############################################
    t_obs_panel = t_obs_panel_list[2]
    ax = ax2
    ax.set_title(f"host halos: t_obs = {t_obs_panel:.1f}")

    for ic, lgm_obs_plot in enumerate(lgm_obs_arr):
        ih = np.argmin(
            np.abs(cendata["t_obs"] - t_obs_panel)
            + np.abs(cendata["lgm_obs"] - lgm_obs_plot)
        )
        tarr = cendata["t_table"][ih]
        mean_log_mah_target = cendata["mean_log_mah"][ih, :]

        lgm_obs = cendata["lgm_obs"][ih]
        t_obs = cendata["t_obs"][ih]
        args = (diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, LGT0_SMDPL)
        mean_log_mah_pred, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(
            *args
        )
        ax.plot(tarr, mean_log_mah_pred - mean_log_mah_target, color=mass_colors[ic])

    ##############################################
    t_obs_panel = t_obs_panel_list[3]
    ax = ax3
    ax.set_title(f"host halos: t_obs = {t_obs_panel:.1f}")

    for ic, lgm_obs_plot in enumerate(lgm_obs_arr):
        ih = np.argmin(
            np.abs(cendata["t_obs"] - t_obs_panel)
            + np.abs(cendata["lgm_obs"] - lgm_obs_plot)
        )
        tarr = cendata["t_table"][ih]
        mean_log_mah_target = cendata["mean_log_mah"][ih, :]

        lgm_obs = cendata["lgm_obs"][ih]
        t_obs = cendata["t_obs"][ih]
        args = (diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, LGT0_SMDPL)
        mean_log_mah_pred, std_log_mah, frac_peaked = mcc.predict_mah_moments_singlebin(
            *args
        )
        ax.plot(tarr, mean_log_mah_pred - mean_log_mah_target, color=mass_colors[ic])

    from matplotlib import lines as mlines

    blue_line = mlines.Line2D([], [], ls="-", c=mblue, label=r"$m_{\rm h}=11.0$")
    red_line = mlines.Line2D([], [], ls="-", c=mred, label=r"$m_{\rm h}=14.5$")
    ax0.legend(handles=[red_line, blue_line], loc="lower left")
    xlabel = ax2.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
    xlabel = ax3.set_xlabel(r"${\rm cosmic\ time\ [Gyr]}$")
    ylabel = ax0.set_ylabel(r"$\Delta\langle M_{\rm h}(t)\rangle\ {\rm [dex]}$")
    ylabel = ax2.set_ylabel(r"$\Delta\langle M_{\rm h}(t)\rangle\ {\rm [dex]}$")

    for ax in ax0, ax1, ax2, ax3:
        xlim = ax.set_xlim(*ax.get_xlim())
        xlim = ax.set_xlim(0.95, xlim[1])
        ax.plot(np.linspace(*xlim, 100), np.zeros(100), "--", color="k")

    os.makedirs(drn, exist_ok=True)
    outname = os.path.join(drn, "diffmahpop_mu_var_4panel_residuals_cens.png")
    fig.savefig(
        outname,
        bbox_extra_artists=[xlabel, ylabel],
        bbox_inches="tight",
        dpi=200,
    )
