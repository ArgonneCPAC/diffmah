"""Model of mean and variance of Mh(t) dMh/dt(t) for Rockstar host halos."""
import numpy as np
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp

LGT0 = 1.137
TODAY = 10 ** LGT0

PARAMS = OrderedDict(
    x0_x0=15.21,
    x0_k=1.68,
    x0_ylo=0.11,
    x0_yhi=1.25,
    k_x0=13.29,
    k_k=-0.40,
    k_ylo=2.51,
    k_yhi=1.34,
    ylo_x0=11.52,
    ylo_k=0.77,
    ylo_ylo=1.18,
    ylo_yhi=3.25,
    yhi_x0=13.43,
    yhi_k=-1.30,
    yhi_ylo=0.97,
    yhi_yhi=0.01,
    tmp_dt=0.02,
    tmp_k=20,
    tmp_indx_t0=4,
)


def get_log_mah(logm0, time, p=np.array(list(PARAMS.values()))):
    """Approximate model for the average Rockstar halo mass across time.

    Parameters
    ----------
    logm0 : float
        Base-10 log of present-day halo mass.

    time : ndarray, shape (n, )
        Cosmic time in Gyr.

    Returns
    -------
    logmh : ndarray, shape (n, )
        Base-10 log of halo mass at the input times.

    """
    return np.array(_get_log_mah_kern(logm0, time, p))


@jjit
def _get_log_mah_kern(logm0, time, p):
    return _global_log_mah_diff_model(p, logm0, time) + logm0


def frac_early_mpeak(logm0):
    """Approximate model for the average Rockstar halo mass across time.

    Parameters
    ----------
    logm0 : ndarray, shape (n, )
        Base-10 log of present-day halo mass.

    Returns
    -------
    f : ndarray, shape (n, )
        Fraction of halos with t_Mpeak < today

    """
    return np.array(_frac_early_mpeak(logm0))


def get_log_mah_scatter(logm0, time):
    """Scatter in halo mass in dex.

    Parameters
    ----------
    logm0 : ndarray, shape (n, )
        Base-10 log of present-day halo mass

    time : ndarray, shape (n, )
        Cosmic time in Gyr.

    Returns
    -------
    scatter : ndarray, shape (n, )
        Scatter in halo mass in dex

    """
    return np.array(_log_mah_scatter(logm0, time))


def get_log_dmhdt(logm0, time, p=np.array(list(PARAMS.values()))):
    """Average mass accretion rate across time.

    Parameters
    ----------
    logm0 : ndarray, shape (n, )
        Base-10 log of present-day halo mass

    time : ndarray, shape (n, )
        Cosmic time in Gyr.

    Returns
    -------
    log_dmhdt : ndarray, shape (n, )
        Base-10 log of dMh/dt in units of Msun/yr

    """
    return np.array(_get_log_dmhdt_kern(logm0, time, p))


@jjit
def _get_log_dmhdt_kern(logm0, time, p):
    specific_log_dmhdt = _specific_mah(logm0, time)
    log_mah = _global_log_mah_diff_model(p, logm0, time) + logm0
    return specific_log_dmhdt + log_mah


def get_log_dmhdt_scatter(logm0, time):
    """Scatter in mass accretion rate across time.

    Parameters
    ----------
    logm0 : ndarray, shape (n, )
        Base-10 log of present-day halo mass

    time : ndarray, shape (n, )
        Cosmic time in Gyr.

    Returns
    -------
    scatter : ndarray, shape (n, )
        Scatter in dMh/dt in dex

    """
    scatter = np.array(_log_dmhdt_scatter(logm0, jnp.log10(time)))
    return scatter


def prob_tmp(
    logm0,
    time,
    today=TODAY,
    tmp_dt=PARAMS["tmp_dt"],
    tmp_k=PARAMS["tmp_k"],
    tmp_indx_t0=PARAMS["tmp_indx_t0"],
):
    """Probability that a halo with mass logm0 reached its peak mass at the input time.

    Parameters
    ----------
    logm0 : ndarray, shape (n, )
        Base-10 log of present-day halo mass

    time : ndarray, shape (n, )
        Cosmic time in Gyr.

    Returns
    -------
    pdf : ndarray, shape (n, )
        Scatter in dMh/dt in dex
    """
    return np.array(_prob_tmp_cens(time, logm0, today, tmp_dt, tmp_k, tmp_indx_t0))


def prob_tmp_early_forming_cens(
    logm0, tmp, t0=TODAY, tmp_k=PARAMS["tmp_k"], tmp_indx_t0=PARAMS["tmp_indx_t0"]
):
    return np.array(_prob_tmp_early_forming_cens(logm0, tmp, t0, tmp_k, tmp_indx_t0))


@jjit
def _global_log_mah_diff_model(params, logmpeak, time):
    x0 = _sigmoid(logmpeak, *params[:4])
    k = _sigmoid(logmpeak, *params[4:8])
    ylo = _sigmoid(logmpeak, *params[8:12])
    yhi = _sigmoid(logmpeak, *params[12:16])
    return _log_mah_diff_model(jnp.log10(time), x0, k, ylo, yhi)


@jjit
def _log_mah_diff_model(lgt, alpha_x0, alpha_k, alpha_ylo, alpha_yhi):
    alpha = _sigmoid(lgt, alpha_x0, alpha_k, alpha_ylo, alpha_yhi)
    return alpha * (lgt - LGT0)


@jjit
def _log_mah_scatter(logm, logt, logt0):
    """Scatter in mh(t) in dex"""
    m = -0.325
    return m * (logt - logt0)


@jjit
def _log_dmhdt_scatter(lgm, lgt):
    """Scatter in dmh/dt(t) in dex"""
    ymin = 0.5
    ymax = _sigmoid(lgm, 12.25, 1.25, 0.25, 1.85)
    x0 = _sigmoid(lgm, 12.35, 3, 0.15, 0.55)
    k = 5
    return _sigmoid(lgt, x0, k, ymin, ymax)


@jjit
def _specific_mah(logm0, time):
    a, b = _specific_mah_params(logm0)
    return a + b * jnp.log10(time)


@jjit
def _specific_mah_params(lgmp):
    a = -8.47
    b = -3.1 + 0.12 * lgmp
    return a, b


@jjit
def _frac_early_mpeak(logm0):
    """Fraction of Rockstar centrals with t_Mpeak < today."""
    return _sigmoid(logm0, 11.7, 1, 0.825, 0.3)


@jjit
def _prob_tmp_cens(tmp, logm0, t0, tmp_dt, tmp_k, tmp_indx_t0):
    """
    """
    frac_early_tmp = _frac_early_mpeak(logm0)

    pdf_tmpeak_early = _prob_tmp_early_forming_cens(logm0, tmp, t0, tmp_k, tmp_indx_t0)

    msk = tmp > t0 - tmp_dt
    return jnp.where(msk, frac_early_tmp, (1 - frac_early_tmp) * pdf_tmpeak_early)


@jjit
def _prob_tmp_early_forming_cens(logm0, tmp, t0, tmp_k, tmp_indx_t0):
    alpha = _prob_tmp_indx(logm0, tmp, tmp_k, tmp_indx_t0)
    pdf_tmpeak_early = _tmp_pdf_powlaw(tmp, alpha, t0)
    return pdf_tmpeak_early


@jjit
def _prob_tmp_indx(logm0, tmp, tmp_k, tmp_indx_t0):
    tmp_x0 = _get_tmp_x0_arr(logm0)
    tmp_falloff = _get_tmp_falloff_arr(logm0)
    alpha = _sigmoid(jnp.log10(tmp), tmp_x0, tmp_k, tmp_falloff, tmp_indx_t0)
    return alpha


@jjit
def _tmp_pdf_powlaw(t, indx, t0):
    x = jnp.where(t > t0, 1, t / t0)
    return (indx / t0) * jnp.power(x, indx - 1)


@jjit
def _get_tmp_x0_arr(logm0):
    return _sigmoid(logm0, 13, 1.5, 0.775, 1)


@jjit
def _get_tmp_falloff_arr(logm0):
    return _sigmoid(logm0, 13.4, 1.6, 4, 15)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))
