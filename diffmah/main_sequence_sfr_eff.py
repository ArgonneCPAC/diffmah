"""Model for time-evolution of SFR efficiency of main sequence centrals,
averaged over halos of the same present-day mass logmp."""
import numpy as np
from collections import OrderedDict
from .utils import jax_sigmoid
from jax import jit as jax_jit
from jax import vmap as jax_vmap


MEAN_SFR_MS_PARAMS = OrderedDict(
    lge0_lgmc=13.41,
    lge0_at_lgmc=-1.1,
    lge0_early_slope=0.46,
    lge0_late_slope=-1.78,
    k_early_x0=11.5,
    k_early_k=3.0,
    k_early_ylo=7.0,
    k_early_yhi=7.0,
    lgtc_x0=12.7,
    lgtc_k=0.56,
    lgtc_ylo=1.68,
    lgtc_yhi=-0.64,
    lgec_x0=12.5,
    lgec_k=13.2,
    lgec_ylo=-0.294,
    lgec_yhi=-0.4,
    k_trans_c0=7.5,
    a_late_x0=15.16,
    a_late_k=2.3,
    a_late_ylo=-0.67,
    a_late_yhi=-3.6,
)

DEFAULT_SFR_MS_PARAMS = OrderedDict(
    lge0=-1.75, k_early=7, lgtc=0.75, lgec=-0.3, k_trans=7.5, a_late=-0.7
)


def mean_log_sfr_efficiency_main_sequence(
    logt,
    logmp,
    lge0_lgmc=MEAN_SFR_MS_PARAMS["lge0_lgmc"],
    lge0_at_lgmc=MEAN_SFR_MS_PARAMS["lge0_at_lgmc"],
    lge0_early_slope=MEAN_SFR_MS_PARAMS["lge0_early_slope"],
    lge0_late_slope=MEAN_SFR_MS_PARAMS["lge0_late_slope"],
    k_early_x0=MEAN_SFR_MS_PARAMS["k_early_x0"],
    k_early_k=MEAN_SFR_MS_PARAMS["k_early_k"],
    k_early_ylo=MEAN_SFR_MS_PARAMS["k_early_ylo"],
    k_early_yhi=MEAN_SFR_MS_PARAMS["k_early_yhi"],
    lgtc_x0=MEAN_SFR_MS_PARAMS["lgtc_x0"],
    lgtc_k=MEAN_SFR_MS_PARAMS["lgtc_k"],
    lgtc_ylo=MEAN_SFR_MS_PARAMS["lgtc_ylo"],
    lgtc_yhi=MEAN_SFR_MS_PARAMS["lgtc_yhi"],
    lgec_x0=MEAN_SFR_MS_PARAMS["lgec_x0"],
    lgec_k=MEAN_SFR_MS_PARAMS["lgec_k"],
    lgec_ylo=MEAN_SFR_MS_PARAMS["lgec_ylo"],
    lgec_yhi=MEAN_SFR_MS_PARAMS["lgec_yhi"],
    k_trans_c0=MEAN_SFR_MS_PARAMS["k_trans_c0"],
    a_late_x0=MEAN_SFR_MS_PARAMS["a_late_x0"],
    a_late_k=MEAN_SFR_MS_PARAMS["a_late_k"],
    a_late_ylo=MEAN_SFR_MS_PARAMS["a_late_ylo"],
    a_late_yhi=MEAN_SFR_MS_PARAMS["a_late_yhi"],
):
    """
    Parameterized model for the star formation efficiency of main sequence centrals
    averaged over halos of the same present-day mass.

    Parameters
    ----------
    logt : ndarray shape (n, )
        Base-10 log of cosmic time in Gyr

    logmp : float
        Base-10 log of peak halo mass at z=0 in units of Msun.

    **params : optional
        Accepts float values for all keyword arguments
        appearing in MEAN_SFR_MS_PARAMS dictionary.

    Returns
    -------
    log_sfr_eff : ndarray shape (n, )
        Base-10 log of SFR efficiency averaged over all main-sequence
        centrals living in halos with present-day mass logmp.

    """

    log_sfr_eff = mean_log_sfr_efficiency_ms_jax(
        logt,
        logmp,
        lge0_lgmc,
        lge0_at_lgmc,
        lge0_early_slope,
        lge0_late_slope,
        k_early_x0,
        k_early_k,
        k_early_ylo,
        k_early_yhi,
        lgtc_x0,
        lgtc_k,
        lgtc_ylo,
        lgtc_yhi,
        lgec_x0,
        lgec_k,
        lgec_ylo,
        lgec_yhi,
        k_trans_c0,
        a_late_x0,
        a_late_k,
        a_late_ylo,
        a_late_yhi,
    )
    return np.array(log_sfr_eff).astype("f4")


def log_sfr_efficiency_main_sequence(
    logt,
    lge0=DEFAULT_SFR_MS_PARAMS["lge0"],
    k_early=DEFAULT_SFR_MS_PARAMS["k_early"],
    lgtc=DEFAULT_SFR_MS_PARAMS["lgtc"],
    lgec=DEFAULT_SFR_MS_PARAMS["lgec"],
    k_trans=DEFAULT_SFR_MS_PARAMS["k_trans"],
    a_late=DEFAULT_SFR_MS_PARAMS["a_late"],
):
    """
    Parameterized model for the star formation efficiency of main sequence centrals.

    Parameters
    ----------
    logt : ndarray shape (n, )
        Base-10 log of cosmic time in Gyr

    logmp : float

    lge0 : float, optional
        Asymptotic value of SFR efficiency at early times

    lgtc : float, optional
        Time of peak star formation in Gyr.

    lgec : float, optional
        Normalization of SFR efficiency at the time of peak SFR.

    a_late : float, optional
        Late-time power-law index of SFR efficiency.

    Default values set in DEFAULT_SFR_MS_PARAMS dictionary.

    Returns
    -------
    log_sfr_eff : ndarray shape (n, )
        Base-10 log of SFR efficiency.

    """
    log_sfr_eff = _log_sfr_efficiency_ms_jax(
        logt, lge0, k_early, lgtc, lgec, k_trans, a_late
    )
    return np.array(log_sfr_eff).astype("f4")


def mean_log_sfr_efficiency_ms_jax(
    logt,
    logmp,
    lge0_lgmc,
    lge0_at_lgmc,
    lge0_early_slope,
    lge0_late_slope,
    k_early_x0,
    k_early_k,
    k_early_ylo,
    k_early_yhi,
    lgtc_x0,
    lgtc_k,
    lgtc_ylo,
    lgtc_yhi,
    lgec_x0,
    lgec_k,
    lgec_ylo,
    lgec_yhi,
    k_trans_c0,
    a_late_x0,
    a_late_k,
    a_late_ylo,
    a_late_yhi,
):
    sfr_eff_params = _get_median_growth_params(
        logmp,
        lge0_lgmc,
        lge0_at_lgmc,
        lge0_early_slope,
        lge0_late_slope,
        k_early_x0,
        k_early_k,
        k_early_ylo,
        k_early_yhi,
        lgtc_x0,
        lgtc_k,
        lgtc_ylo,
        lgtc_yhi,
        lgec_x0,
        lgec_k,
        lgec_ylo,
        lgec_yhi,
        k_trans_c0,
        a_late_x0,
        a_late_k,
        a_late_ylo,
        a_late_yhi,
    )
    return _log_sfr_efficiency_ms_jax(logt, *sfr_eff_params)


def _log_sfr_efficiency_ms_jax_kern(logt, lge0, k_early, lgtc, lgec, k_trans, a_late):
    dy = lgec - lge0
    epsilon_early = jax_sigmoid(logt, lgtc, k_early, lge0, lge0 + 2 * dy)
    epsilon_late = a_late * (logt - lgtc) + lgec - 1 / 4
    return jax_sigmoid(logt, lgtc, k_trans, epsilon_early, epsilon_late)


_log_sfr_efficiency_ms_jax = jax_jit(
    jax_vmap(
        _log_sfr_efficiency_ms_jax_kern, in_axes=(0, None, None, None, None, None, None)
    )
)


def _get_median_growth_params(
    logm,
    lge0_lgmc,
    lge0_at_lgmc,
    lge0_early_slope,
    lge0_late_slope,
    k_early_x0,
    k_early_k,
    k_early_ylo,
    k_early_yhi,
    lgtc_x0,
    lgtc_k,
    lgtc_ylo,
    lgtc_yhi,
    lgec_x0,
    lgec_k,
    lgec_ylo,
    lgec_yhi,
    k_trans_c0,
    a_late_x0,
    a_late_k,
    a_late_ylo,
    a_late_yhi,
):
    lge0 = _lge0_vs_lgm0_kern(
        logm, lge0_lgmc, lge0_at_lgmc, lge0_early_slope, lge0_late_slope
    )
    k_early = _k_early_vs_lgm0_kern(
        logm, k_early_x0, k_early_k, k_early_ylo, k_early_yhi
    )
    lgtc = _lgtc_vs_lgm0_kern(logm, lgtc_x0, lgtc_k, lgtc_ylo, lgtc_yhi)
    lgec = _lgec_vs_lgm0_kern(logm, lgec_x0, lgec_k, lgec_ylo, lgec_yhi)
    k_trans = _k_trans_vs_lgm0_kern(logm, k_trans_c0)
    a_late = _a_late_vs_lgm0_kern(logm, a_late_x0, a_late_k, a_late_ylo, a_late_yhi)
    return lge0, k_early, lgtc, lgec, k_trans, a_late


def _lge0_vs_lgm0_kern(
    logm, lge0_lgmc, lge0_at_lgmc, lge0_early_slope, lge0_late_slope
):
    ylo = lge0_at_lgmc + (logm - lge0_lgmc) * lge0_early_slope
    yhi = lge0_at_lgmc + (logm - lge0_lgmc) * lge0_late_slope
    return jax_sigmoid(logm, lge0_lgmc, 5, ylo, yhi)


def _k_early_vs_lgm0_kern(logm, k_early_x0, k_early_k, k_early_ylo, k_early_yhi):
    return jax_sigmoid(logm, k_early_x0, k_early_k, k_early_ylo, k_early_yhi)


def _lgtc_vs_lgm0_kern(logm, lgtc_x0, lgtc_k, lgtc_ylo, lgtc_yhi):
    return jax_sigmoid(logm, lgtc_x0, lgtc_k, lgtc_ylo, lgtc_yhi)


def _lgec_vs_lgm0_kern(logm, lgec_x0, lgec_k, lgec_ylo, lgec_yhi):
    return jax_sigmoid(logm, lgec_x0, lgec_k, lgec_ylo, lgec_yhi)


def _k_trans_vs_lgm0_kern(logm, k_trans_c0):
    return logm - logm + k_trans_c0


def _a_late_vs_lgm0_kern(logm, a_late_x0, a_late_k, a_late_ylo, a_late_yhi):
    return jax_sigmoid(logm, a_late_x0, a_late_k, a_late_ylo, a_late_yhi)
