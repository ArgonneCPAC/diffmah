"""
"""
import numpy as np
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import grad
from jax import vmap as jvmap
from jax.scipy.stats import multivariate_normal as jnorm
from .utils import get_1d_arrays


MAH_PARAMS = OrderedDict(
    mah_x0=-0.15, mah_k=4.0, mah_early_index=3.0, mah_late_index=1.0
)

MEAN_MAH_U_PARAMS = OrderedDict(
    log_early_index_x0=14.315,
    log_early_index_log_k=-0.381,
    log_early_index_ylo=-0.038,
    log_early_index_yhi=1.119,
    u_dy_x0=12.932,
    u_dy_log_k=-0.085,
    u_dy_ylo=-0.518,
    u_dy_yhi=0.147,
    index_cov_x0=14.393,
    index_cov_log_k=0.260,
    index_cov_u_lo=-10.648,
    index_cov_u_hi=4.900,
    index_cov_v_lo=-3.053,
    index_cov_v_hi=-2.403,
    index_cov_c_lo=-0.592,
    index_cov_c_hi=-7.443,
)


TMP_PDF_PARAMS = OrderedDict(tmp_k=20.0, tmp_indx_t0=4.0)
TODAY = 13.8


@jjit
def _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, log_early_index, u_dy):
    """
    """
    early_index, late_index = _get_bounded_params(log_early_index, u_dy)
    return _rolling_plaw(logt, logtmp, logmp, x0, k, early_index, late_index)


@jjit
def _rolling_plaw(logt, logtmp, logmp, x0, k, ymin, ymax):
    p = _sigmoid(logt, x0, k, ymin, ymax)
    log_mah = p * (logt - logtmp) + logmp
    return log_mah


@jjit
def _get_bounded_params(log_early_index, u_dy):
    early_index = 10 ** log_early_index
    delta_index_max = early_index
    delta_index = _sigmoid(u_dy, 0, 1, 0, delta_index_max)
    late_index = early_index - delta_index
    return early_index, late_index


@jjit
def _get_unbounded_params(early_index, late_index):
    log_early_index = jnp.log10(early_index)
    delta_index_max = early_index
    delta_index = early_index - late_index
    u_dy = _inverse_sigmoid(delta_index, 0, 1, 0, delta_index_max)
    return log_early_index, u_dy


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


def _calc_mah(time, logtmp, logmp, x0, k, log_early_index, u_dy):
    logt = jnp.log10(time)
    return 10 ** _rolling_plaw_bounded(
        logt, logtmp, logmp, x0, k, log_early_index, u_dy
    )


@jjit
def _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy):
    log_mah = _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    return jnp.where(log_mah > logmp, logmp, log_mah)


@jjit
def _calc_clipped_mah(t, lgtmp, logmp, x0, k, log_early_index, u_dy):
    logt = jnp.log10(t)
    return 10 ** _calc_clipped_log_mah(logt, lgtmp, logmp, x0, k, log_early_index, u_dy)


_calc_dmhdt = jjit(jvmap(grad(_calc_mah, argnums=0), in_axes=(0, *[None] * 6)))
_calc_clipped_dmhdt = jjit(
    jvmap(grad(_calc_clipped_mah, argnums=0), in_axes=(0, *[None] * 6))
)


@jjit
def _get_clipped_log_mah_kern(logt, logtmp, k, logmp, x0, log_early_index, u_dy):
    log_mah = _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    dmhdt = _calc_clipped_dmhdt(10 ** logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    return log_mah, dmhdt / 1e9


@jjit
def _get_clipped_sigmah_kern(logt, logtmp, k, logmp, x0, log_early_index, u_dy):
    log_mah = _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    dmhdt = _calc_clipped_dmhdt(10 ** logt, logtmp, logmp, x0, k, log_early_index, u_dy)
    return 10 ** log_mah, dmhdt / 1e9


@jjit
def _get_sigmah_halopop_kern(logt, logtmp, k, logmp, x0, early_index, late_index):
    log_early_index, u_dy = _get_unbounded_params(early_index, late_index)
    return _get_clipped_sigmah_kern(logt, logtmp, k, logmp, x0, log_early_index, u_dy)


_f0 = jvmap(_get_clipped_sigmah_kern, in_axes=(*[None] * 3, 0, *[None] * 3))
_f1 = jvmap(_f0, in_axes=(None, 0, *[None] * 5))
_f2 = jvmap(_f1, in_axes=(*[None] * 5, 0, None))
_get_avg_mah_early_tmp_integrand = jjit(jvmap(_f2, in_axes=(*[None] * 5, None, 0)))


_g0 = jvmap(_get_clipped_sigmah_kern, in_axes=(*[None] * 3, 0, *[None] * 3))
_g1 = jvmap(_g0, in_axes=(*[None] * 5, 0, None))
_get_avg_mah_tmp_today_integrand = jjit(jvmap(_g1, in_axes=(*[None] * 5, None, 0)))


@jjit
def _mean_log_early_index(
    lgmp,
    log_early_index_x0,
    log_early_index_log_k,
    log_early_index_ylo,
    log_early_index_yhi,
):
    return _sigmoid(
        lgmp,
        log_early_index_x0,
        10 ** log_early_index_log_k,
        log_early_index_ylo,
        log_early_index_yhi,
    )


@jjit
def _mean_u_dy(lgmp, u_dy_x0, u_dy_log_k, u_dy_ylo, u_dy_yhi):
    return _sigmoid(lgmp, u_dy_x0, 10 ** u_dy_log_k, u_dy_ylo, u_dy_yhi)


@jjit
def _mean_early_index(lgmp, early_x0, early_log_k, early_ylo, early_yhi):
    return _sigmoid(lgmp, early_x0, 10 ** early_log_k, early_ylo, early_yhi)


def mean_u_dy(
    lgmp,
    u_dy_x0=MEAN_MAH_U_PARAMS["u_dy_x0"],
    u_dy_log_k=MEAN_MAH_U_PARAMS["u_dy_log_k"],
    u_dy_ylo=MEAN_MAH_U_PARAMS["u_dy_ylo"],
    u_dy_yhi=MEAN_MAH_U_PARAMS["u_dy_yhi"],
):
    return np.array(_mean_u_dy(lgmp, u_dy_x0, u_dy_log_k, u_dy_ylo, u_dy_yhi))


def mean_log_early_index(
    lgmp,
    early_x0=MEAN_MAH_U_PARAMS["log_early_index_x0"],
    early_log_k=MEAN_MAH_U_PARAMS["log_early_index_log_k"],
    early_ylo=MEAN_MAH_U_PARAMS["log_early_index_ylo"],
    early_yhi=MEAN_MAH_U_PARAMS["log_early_index_yhi"],
):
    return np.array(
        _mean_early_index(lgmp, early_x0, early_log_k, early_ylo, early_yhi)
    )


def early_index_u_dy_covariance(
    logmpeak,
    index_cov_x0=MEAN_MAH_U_PARAMS["index_cov_x0"],
    index_cov_log_k=MEAN_MAH_U_PARAMS["index_cov_log_k"],
    index_cov_u_lo=MEAN_MAH_U_PARAMS["index_cov_u_lo"],
    index_cov_v_lo=MEAN_MAH_U_PARAMS["index_cov_v_lo"],
    index_cov_c_lo=MEAN_MAH_U_PARAMS["index_cov_c_lo"],
    index_cov_u_hi=MEAN_MAH_U_PARAMS["index_cov_u_hi"],
    index_cov_v_hi=MEAN_MAH_U_PARAMS["index_cov_v_hi"],
    index_cov_c_hi=MEAN_MAH_U_PARAMS["index_cov_c_hi"],
):
    index_cov_u = _sigmoid(
        logmpeak, index_cov_x0, 10 ** index_cov_log_k, index_cov_u_lo, index_cov_u_hi
    )
    index_cov_v = _sigmoid(
        logmpeak, index_cov_x0, 10 ** index_cov_log_k, index_cov_v_lo, index_cov_v_hi
    )
    index_cov_c = _sigmoid(
        logmpeak, index_cov_x0, 10 ** index_cov_log_k, index_cov_c_lo, index_cov_c_hi
    )
    return np.array(_2d_covariance(index_cov_u, index_cov_v, index_cov_c))


@jjit
def _early_index_u_dy_covariance_kern(
    logmpeak,
    index_cov_x0,
    index_cov_log_k,
    index_cov_u_lo,
    index_cov_v_lo,
    index_cov_c_lo,
    index_cov_u_hi,
    index_cov_v_hi,
    index_cov_c_hi,
):
    index_cov_u = _sigmoid(
        logmpeak, index_cov_x0, 10 ** index_cov_log_k, index_cov_u_lo, index_cov_u_hi
    )
    index_cov_v = _sigmoid(
        logmpeak, index_cov_x0, 10 ** index_cov_log_k, index_cov_v_lo, index_cov_v_hi
    )
    index_cov_c = _sigmoid(
        logmpeak, index_cov_x0, 10 ** index_cov_log_k, index_cov_c_lo, index_cov_c_hi
    )
    return _2d_covariance(index_cov_u, index_cov_v, index_cov_c)


_early_index_u_dy_covariance = jjit(
    jvmap(_early_index_u_dy_covariance_kern, in_axes=(0, *[None] * 8))
)


@jjit
def _2d_covariance(u, v, c):
    """Bijection between R^3 and the space of
    2d positive-definite covariance matrices."""
    _x = jnp.sqrt(1.0 + u * u + v * v)
    cov = jnp.exp(c) * jnp.array(((_x + u, v), (v, _x - u)))
    return cov


_get_sigmah_halopop = jjit(jvmap(_get_sigmah_halopop_kern, in_axes=(None, *[0] * 6)))


def individual_halo_assembly_history(
    cosmic_time,
    logmp,
    tmp=TODAY,
    mah_x0=MAH_PARAMS["mah_x0"],
    mah_k=MAH_PARAMS["mah_k"],
    mah_early_index=None,
    mah_late_index=None,
):
    """Rolling power-law model for the mass assembly of individual halos.

    Parameters
    ----------
    cosmic_time : ndarray of shape (n_times, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    logmp : float or ndarray of shape (n_mass, )
        Base-10 log of peak halo mass in units of Msun

    tmp : float or ndarray of shape (n_mass, ), optional
        Age of the universe in Gyr at the time halo mass attains the input logmp.

    mah_x0 : float or ndarray of shape (n_mass, ), optional
        Base-10 log of the time of peak star formation.

    mah_k : float or ndarray of shape (n_mass, ), optional
        Transition speed between early- and late-time power laws indices.

    mah_early_index : float or ndarray of shape (n_mass, ), optional
        Early-time power-law index Mh ~ (t/tmp)**mah_early_index for logt << mah_x0.

    mah_late_index : float or ndarray of shape (n_mass, ), optional
        Late-time power-law index Mh ~ t**mah_late_index for logt >> mah_x0.

    Returns
    -------
    log_mah : ndarray of shape (n_mass, n_times)
        Base-10 log of halo mass at the input times.
        Halo mass is in units of Msun.
        If n_mass = 1, log_mah will be flattened to have shape (n_times, )

    log_dmhdt : ndarray of shape (n_mass, n_times)
        Base-10 log of halo mass accretion rate at the input times
        Accretion rate is in units of Msun/yr.
        If n_mass = 1, log_dmhdt will be flattened to have shape (n_times, )

    """
    lgt, logmp, lgtmp, x0, k, early, late = _process_args(
        cosmic_time, logmp, tmp, mah_x0, mah_k, mah_early_index, mah_late_index
    )
    logmah, log_dmhdt = _get_sigmah_halopop(lgt, lgtmp, k, logmp, x0, early, late)

    n_mass = logmah.shape[0]
    if n_mass == 1:
        return np.array(logmah).flatten(), np.array(log_dmhdt).flatten()
    else:
        return np.array(logmah), np.array(log_dmhdt)


def _process_args(t, logmp, tmp, x0, k, early, late):
    logt = np.atleast_1d(np.log10(t))
    logmp = np.atleast_1d(logmp)
    logtmp = np.log10(tmp)

    if early is None:
        u_p = [
            MEAN_MAH_U_PARAMS[key] for key in MEAN_MAH_U_PARAMS.keys() if "early" in key
        ]
        mean_log_early = _mean_log_early_index(logmp, *u_p)
        early = 10.0 ** mean_log_early

    if late is None:
        u_dy_p = [
            MEAN_MAH_U_PARAMS[key] for key in MEAN_MAH_U_PARAMS.keys() if "u_dy" in key
        ]
        mean_u_dy = _mean_u_dy(logmp, *u_dy_p)
        late = _get_bounded_params(np.log10(early), mean_u_dy)[1]

    logmp, logtmp, x0, k, early, late = get_1d_arrays(logmp, logtmp, x0, k, early, late)
    return logt, logmp, logtmp, x0, k, early, late


def _g0(a, b, mu, cov):
    x = jnp.array((a, b)).astype("f4")
    return jnorm.pdf(x, mu, cov)


_g1 = jvmap(_g0, in_axes=(None, None, 0, 0))
_g2 = jvmap(_g1, in_axes=(0, None, None, None))
_g3 = jvmap(_g2, in_axes=(None, 0, None, None))


@jjit
def _get_index_weights_kern(log10_early_index, u_dy, mu_arr, cov_arr):
    _pdf = _g3(log10_early_index, u_dy, mu_arr, cov_arr)
    n_mass = _pdf.shape[2]
    _norm = jnp.sum(_pdf, axis=(0, 1))
    _norm = jnp.where(_norm == 0, 1.0, _norm)
    pdf = _pdf / _norm.reshape((1, 1, n_mass))
    return pdf


@jjit
def _get_mean_index_weights(
    logmparr,
    log10_early_arr,
    u_dy_arr,
    log_early_index_x0,
    log_early_index_log_k,
    log_early_index_ylo,
    log_early_index_yhi,
    u_dy_x0,
    u_dy_log_k,
    u_dy_ylo,
    u_dy_yhi,
    index_cov_x0,
    index_cov_log_k,
    index_cov_u_lo,
    index_cov_v_lo,
    index_cov_c_lo,
    index_cov_u_hi,
    index_cov_v_hi,
    index_cov_c_hi,
):
    mean_u_dy = _mean_u_dy(logmparr, u_dy_x0, u_dy_log_k, u_dy_ylo, u_dy_yhi)
    p = (
        log_early_index_x0,
        log_early_index_log_k,
        log_early_index_ylo,
        log_early_index_yhi,
    )
    log_mean_early = _mean_log_early_index(logmparr, *p,)

    X = jnp.array((log_mean_early, mean_u_dy)).T
    cov_arr = _early_index_u_dy_covariance(
        logmparr,
        index_cov_x0,
        index_cov_log_k,
        index_cov_u_lo,
        index_cov_v_lo,
        index_cov_c_lo,
        index_cov_u_hi,
        index_cov_v_hi,
        index_cov_c_hi,
    )
    mean_index_weights = _get_index_weights_kern(log10_early_arr, u_dy_arr, X, cov_arr)
    return mean_index_weights


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
def _get_tmp_x0_arr(logm0):
    return _sigmoid(logm0, 13, 1.5, 0.775, 1)


@jjit
def _get_tmp_falloff_arr(logm0):
    return _sigmoid(logm0, 13.4, 1.6, 4, 15)


@jjit
def _tmp_pdf_powlaw(t, indx, t0):
    x = jnp.where(t > t0, 1, t / t0)
    return (indx / t0) * jnp.power(x, indx - 1)


_h0 = jvmap(_prob_tmp_early_forming_cens, in_axes=(None, 0, None, None, None))


@jjit
def _h1(logm0, tmparr, t0, tmp_k, tmp_indx_t0):
    _pdf = _h0(logm0, tmparr, t0, tmp_k, tmp_indx_t0)
    pdf = _pdf / _pdf.sum()
    return pdf


_get_tmp_weights_kern = jjit(jvmap(_h1, in_axes=(0, *[None] * 4)))


@jjit
def get_tmp_weights_kern(logm0, tmparr, t0, tmp_k, tmp_indx_t0):
    return _get_tmp_weights_kern(logm0, tmparr, t0, tmp_k, tmp_indx_t0).T


@jjit
def _frac_early_mpeak(logmp):
    """Fraction of Rockstar centrals with t_Mpeak < today."""
    return _sigmoid(logmp, 11.7, 1, 0.825, 0.3)


@jjit
def _weighted_history_early_tmp_halos(
    logt,
    logtmparr,
    k,
    logmparr,
    x0,
    log10_early_arr,
    u_dy_arr,
    log_early_index_x0,
    log_early_index_log_k,
    log_early_index_ylo,
    log_early_index_yhi,
    u_dy_x0,
    u_dy_log_k,
    u_dy_ylo,
    u_dy_yhi,
    index_cov_x0,
    index_cov_log_k,
    index_cov_u_lo,
    index_cov_v_lo,
    index_cov_c_lo,
    index_cov_u_hi,
    index_cov_v_hi,
    index_cov_c_hi,
    tmp_k,
    tmp_indx_t0,
    today,
):
    mah_integrand, dmhdt_integrand = _get_avg_mah_early_tmp_integrand(
        logt, logtmparr, k, logmparr, x0, log10_early_arr, u_dy_arr
    )
    n_late, n_early, n_tmp, n_mass, n_times = mah_integrand.shape

    index_weights = _get_mean_index_weights(
        logmparr,
        log10_early_arr,
        u_dy_arr,
        log_early_index_x0,
        log_early_index_log_k,
        log_early_index_ylo,
        log_early_index_yhi,
        u_dy_x0,
        u_dy_log_k,
        u_dy_ylo,
        u_dy_yhi,
        index_cov_x0,
        index_cov_log_k,
        index_cov_u_lo,
        index_cov_v_lo,
        index_cov_c_lo,
        index_cov_u_hi,
        index_cov_v_hi,
        index_cov_c_hi,
    )

    tmp_weights = get_tmp_weights_kern(
        logmparr, 10 ** logtmparr, today, tmp_k, tmp_indx_t0
    )
    _w0 = index_weights.reshape((n_late, n_early, 1, n_mass, 1))
    _w1 = tmp_weights.reshape((1, 1, n_tmp, n_mass, 1))
    weights = _w0 * _w1

    return mah_integrand, dmhdt_integrand, weights


@jjit
def _weighted_history_tmp_today_halos(
    logt,
    k,
    logmparr,
    x0,
    log10_early_arr,
    u_dy_arr,
    log_early_index_x0,
    log_early_index_log_k,
    log_early_index_ylo,
    log_early_index_yhi,
    u_dy_x0,
    u_dy_log_k,
    u_dy_ylo,
    u_dy_yhi,
    index_cov_x0,
    index_cov_log_k,
    index_cov_u_lo,
    index_cov_v_lo,
    index_cov_c_lo,
    index_cov_u_hi,
    index_cov_v_hi,
    index_cov_c_hi,
    tmp_k,
    tmp_indx_t0,
    today,
):
    mah_integrand, dmhdt_integrand = _get_avg_mah_tmp_today_integrand(
        logt, jnp.log10(today), k, logmparr, x0, log10_early_arr, u_dy_arr
    )
    n_late, n_early, n_mass, n_times = mah_integrand.shape

    index_weights = _get_mean_index_weights(
        logmparr,
        log10_early_arr,
        u_dy_arr,
        log_early_index_x0,
        log_early_index_log_k,
        log_early_index_ylo,
        log_early_index_yhi,
        u_dy_x0,
        u_dy_log_k,
        u_dy_ylo,
        u_dy_yhi,
        index_cov_x0,
        index_cov_log_k,
        index_cov_u_lo,
        index_cov_v_lo,
        index_cov_c_lo,
        index_cov_u_hi,
        index_cov_v_hi,
        index_cov_c_hi,
    )

    weights = index_weights.reshape((n_late, n_early, n_mass, 1))
    return mah_integrand, dmhdt_integrand, weights


def _integrate_halos(histories, weights, axis):
    return jnp.sum(weights * histories, axis=axis)


@jjit
def _calc_avg_halo_history(
    logt,
    logtmparr,
    k,
    logmparr,
    x0,
    log10_early_arr,
    u_dy_arr,
    log_early_index_x0,
    log_early_index_log_k,
    log_early_index_ylo,
    log_early_index_yhi,
    u_dy_x0,
    u_dy_log_k,
    u_dy_ylo,
    u_dy_yhi,
    index_cov_x0,
    index_cov_log_k,
    index_cov_u_lo,
    index_cov_v_lo,
    index_cov_c_lo,
    index_cov_u_hi,
    index_cov_v_hi,
    index_cov_c_hi,
    tmp_k,
    tmp_indx_t0,
    today,
):
    _res_early = _weighted_history_early_tmp_halos(
        logt,
        logtmparr,
        k,
        logmparr,
        x0,
        log10_early_arr,
        u_dy_arr,
        log_early_index_x0,
        log_early_index_log_k,
        log_early_index_ylo,
        log_early_index_yhi,
        u_dy_x0,
        u_dy_log_k,
        u_dy_ylo,
        u_dy_yhi,
        index_cov_x0,
        index_cov_log_k,
        index_cov_u_lo,
        index_cov_v_lo,
        index_cov_c_lo,
        index_cov_u_hi,
        index_cov_v_hi,
        index_cov_c_hi,
        tmp_k,
        tmp_indx_t0,
        today,
    )
    mah_int_early, dmhdt_int_early, w_early = _res_early
    smar_int_early = 1e9 * dmhdt_int_early / mah_int_early

    _res_t0 = _weighted_history_tmp_today_halos(
        logt,
        k,
        logmparr,
        x0,
        log10_early_arr,
        u_dy_arr,
        log_early_index_x0,
        log_early_index_log_k,
        log_early_index_ylo,
        log_early_index_yhi,
        u_dy_x0,
        u_dy_log_k,
        u_dy_ylo,
        u_dy_yhi,
        index_cov_x0,
        index_cov_log_k,
        index_cov_u_lo,
        index_cov_v_lo,
        index_cov_c_lo,
        index_cov_u_hi,
        index_cov_v_hi,
        index_cov_c_hi,
        tmp_k,
        tmp_indx_t0,
        today,
    )
    mah_int_t0, dmhdt_int_t0, w_t0 = _res_t0
    smar_int_t0 = 1e9 * dmhdt_int_t0 / mah_int_t0

    n_mass = mah_int_t0.shape[0]

    n_late, n_early, n_tmp, n_mass, n_times = mah_int_early.shape

    avg_mah_early = _integrate_halos(mah_int_early, w_early, (0, 1, 2))
    avg_dmhdt_early = _integrate_halos(dmhdt_int_early, w_early, (0, 1, 2))
    avg_smar_early = _integrate_halos(smar_int_early, w_early, (0, 1, 2))

    avg_mah_t0 = _integrate_halos(mah_int_t0, w_t0, (0, 1))
    avg_dmhdt_t0 = _integrate_halos(dmhdt_int_t0, w_t0, (0, 1))
    avg_smar_t0 = _integrate_halos(smar_int_t0, w_t0, (0, 1))

    F_e = _frac_early_mpeak(logmparr).reshape((n_mass, 1))

    avg_mah = F_e * avg_mah_early + (1 - F_e) * avg_mah_t0
    avg_dmhdt = F_e * avg_dmhdt_early + (1 - F_e) * avg_dmhdt_t0
    avg_smar = F_e * avg_smar_early + (1 - F_e) * avg_smar_t0

    log_mah_int_early = jnp.log10(mah_int_early)
    log_mah_int_t0 = jnp.log10(mah_int_t0)
    log_avg_mah = jnp.log10(avg_mah)
    d_log_mah_int_early = log_mah_int_early - log_avg_mah
    d_log_mah_int_t0 = log_mah_int_t0 - log_avg_mah

    var_mah_early = jnp.sum(w_early * (d_log_mah_int_early ** 2), axis=(0, 1, 2))
    var_mah_t0 = jnp.sum(w_t0 * (d_log_mah_int_t0 ** 2), axis=(0, 1))
    var_mah = F_e * var_mah_early + (1 - F_e) * var_mah_t0

    return avg_mah, avg_dmhdt, avg_smar, var_mah


@jjit
def calc_avg_halo_history(params, data):
    args = (*data[0:7], *params, *data[7:])
    return _calc_avg_halo_history(*args)


@jjit
def _mse(pred, target):
    err = (pred - target) / target
    return jnp.mean(err * err)


_mse_bundle = jvmap(_mse, in_axes=(0, 0))


@jjit
def mse_bundle(preds, targets):
    return jnp.sum(_mse_bundle(preds, targets))


@jjit
def _avg_halo_history_loss(params, loss_data):
    target_halo_mahs = loss_data[-1]
    pred_data = loss_data[0:-1]
    avg_mah, avg_dmhdt, avg_smar, var_smar = calc_avg_halo_history(params, pred_data)
    return mse_bundle(avg_mah, target_halo_mahs)


@jjit
def _avg_mah_std_dmhdt_loss(params, loss_data):
    target_halo_mahs, target_halo_smars = loss_data[-2:]
    pred_data = loss_data[0:-2]
    avg_mah, avg_dmhdt, avg_smar, var_mah = calc_avg_halo_history(params, pred_data)
    avg_mah_loss = mse_bundle(avg_mah, target_halo_mahs)
    avg_smar_loss = mse_bundle(avg_smar, target_halo_smars)
    # var_mah_loss = mse_bundle(var_mah, target_halo_var_mah)
    return avg_mah_loss + avg_smar_loss  # + var_mah_loss
