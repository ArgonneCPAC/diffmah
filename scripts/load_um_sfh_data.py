"""Functions loading the halos from either BPl or MDPL into memory."""
import numpy as np
import os
from astropy.table import Table
import warnings
from numba import jit as numba_jit
from diffmah.halo_assembly import _get_dt_array

MDPL2_DRN = "/Users/aphearin/work/DATA/MOCKS/UniverseMachine/um_dr1_history_samples"
BPL_DRN = "/Users/aphearin/work/DATA/MOCKS/UniverseMachine/SFH_samples/"

LOG_SSFR_CLIP = -11.0


def load_bpl_data(drn=BPL_DRN, log_ssfr_clip=LOG_SSFR_CLIP):
    """
    """
    bpl = Table(np.load(os.path.join(drn, "um_histories_dr1_bpl_cens_a_1.002310.npy")))
    bpl_t = np.loadtxt(os.path.join(drn, "cosmic_times_bpl.dat"))
    bpl_z = 1 / np.loadtxt(os.path.join(drn, "scale_list_bpl.dat")) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mah = np.log10(
            np.maximum.accumulate(bpl["mpeak_history_main_prog"], axis=1)
        )
        bpl["log_mah"] = np.where(np.isfinite(log_mah), log_mah, 0)
    bpl.remove_column("mpeak_history_main_prog")

    nh, nt = bpl["log_mah"].shape
    tmparr = np.zeros(nh)
    for i, log_mah in enumerate(bpl["log_mah"]):
        tmparr[i] = bpl_t[find_indx_xpeak(log_mah, nt)]
    bpl["tmp"] = tmparr

    nh, nt = bpl["log_mah"].shape
    dmhdt_matrix = np.zeros((nh, nt))
    for i, log_mah in enumerate(bpl["log_mah"]):
        out = np.zeros(nt)
        calculate_dmhdt(bpl_t, 10 ** log_mah, out, nt)
        dmhdt_matrix[i, :] = out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bpl["log_dmhdt"] = np.where(dmhdt_matrix <= 0, 0, np.log10(dmhdt_matrix))

    sfrh = bpl["sfr_history_main_prog"]
    dtarr = _get_dt_array(bpl_t)
    smh, log_ssfrh = _get_clipped_sfh_samples(bpl_t, dtarr, sfrh, log_ssfr_clip)
    bpl["smh"] = smh
    bpl["log_ssfrh"] = log_ssfrh

    return bpl, bpl_t, bpl_z


def load_mdpl2_data(drn=MDPL2_DRN, log_ssfr_clip=LOG_SSFR_CLIP):
    """
    """
    mdpl2 = Table(np.load(os.path.join(drn, "um_histories_dr1_mdpl2_cens.npy")))
    mdpl_t = np.loadtxt(os.path.join(drn, "mdpl2_cosmic_time.txt"))
    mdpl_z = 1 / np.loadtxt(os.path.join(drn, "mdpl2_scale_factors.txt")) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mah = np.log10(
            np.maximum.accumulate(mdpl2["mpeak_history_main_prog"], axis=1)
        )
        mdpl2["log_mah"] = np.where(np.isfinite(log_mah), log_mah, 0)

    mdpl2.remove_column("mpeak_history_main_prog")

    nh, nt = mdpl2["log_mah"].shape
    tmparr = np.zeros(nh)
    for i, log_mah in enumerate(mdpl2["log_mah"]):
        tmparr[i] = mdpl_t[find_indx_xpeak(log_mah, nt)]
    mdpl2["tmp"] = tmparr

    nh, nt = mdpl2["log_mah"].shape
    dmhdt_matrix = np.zeros((nh, nt))
    for i, log_mah in enumerate(mdpl2["log_mah"]):
        out = np.zeros(nt)
        calculate_dmhdt(mdpl_t, 10 ** log_mah, out, nt)
        dmhdt_matrix[i, :] = out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdpl2["log_dmhdt"] = np.where(dmhdt_matrix <= 0, 0, np.log10(dmhdt_matrix))

    sfrh = mdpl2["sfr_history_main_prog"]
    dtarr = _get_dt_array(mdpl_t)
    smh, log_ssfrh = _get_clipped_sfh_samples(mdpl_t, dtarr, sfrh, log_ssfr_clip)
    mdpl2["smh"] = smh
    mdpl2["log_ssfrh"] = log_ssfrh

    return mdpl2, mdpl_t, mdpl_z


@numba_jit
def find_indx_xpeak(x, n):
    """Find the index where x first attains its peak value."""
    xpeak = x[0]
    indx_xpeak = 0
    for i in range(1, n):
        x_i = x[i]
        if x_i > xpeak:
            xpeak = x_i
            indx_xpeak = i
    return indx_xpeak


@numba_jit
def calculate_dmhdt(tarr, marr, dmdht_out, n):
    """Calculate mass accretion rate history from mass history.
    Assumes marr is monotonically increasing.

    Parameters
    ----------
    tarr : ndarray of shape (n, )
        Cosmic time in Gyr

    marr : ndarray of shape (n, )
        Halo mass in Msun. Algorithm assumes marr is monotonically increasing.

    Returns
    -------
    dmhdt : ndarray of shape (n, )
        Mass accretion rate in Msun/yr

    """
    lgmarr = np.log10(marr)

    dt_init = tarr[1] - tarr[0]
    dm_init = marr[1] - marr[0]
    dmdt_init = dm_init / dt_init
    dmdht_out[0] = dmdt_init

    for i in range(1, n - 1):
        dt_lo = (tarr[i] - tarr[i - 1]) / 2
        dt_hi = (tarr[i + 1] - tarr[i]) / 2
        tlo = tarr[i] - dt_lo
        thi = tarr[i] + dt_hi
        dt = thi - tlo

        lgm = lgmarr[i]
        lgm_lo = lgmarr[i - 1]
        lgm_hi = lgmarr[i + 1]
        lgm_lo = (lgm + lgm_lo) / 2
        lgm_hi = (lgm_hi + lgm) / 2
        dm = 10 ** lgm_hi - 10 ** lgm_lo

        dmdht_out[i] = dm / dt / 1e9

    dt_final = tarr[n - 1] - tarr[n - 2]
    dm_final = marr[n - 1] - marr[n - 2]
    dmdt_final = dm_final / dt_final
    dmdht_out[n - 1] = dmdt_final / 1e9


def _get_clipped_sfh_samples(tarr, dtarr, sfrh, log_ssfr_clip):
    """
    """
    ssfr_clip = 10 ** log_ssfr_clip
    smh_samples = np.cumsum(sfrh * dtarr * 1e9, axis=1)
    msk_nonzero = (smh_samples > 0) & np.isfinite(smh_samples)
    _ssfrh = sfrh / np.where(msk_nonzero, smh_samples, 1.0)
    msk_clip = ~msk_nonzero | (_ssfrh < ssfr_clip)
    log_ssfrh_samples = np.log10(np.where(msk_clip, ssfr_clip, _ssfrh))
    return smh_samples, log_ssfrh_samples
