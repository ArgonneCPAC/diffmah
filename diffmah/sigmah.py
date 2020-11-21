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


BOUNDS = OrderedDict(mah_early_index=(0.0, 10.0))
MAH_PARAMS = OrderedDict(
    mah_x0=-0.15, mah_k=4.0, mah_early_index=3.0, mah_late_index=1.0
)
MEAN_MAH_PARAMS = OrderedDict(
    early_index_x0=14.38,
    early_index_k=0.6,
    early_index_ylo=2.0,
    early_index_yhi=6.1,
    late_index_x0=15.1,
    late_index_k=0.75,
    late_index_ylo=0.59,
    late_index_yhi=3.47,
    log_mah_det_cov=-2.38,
)

TMP_PDF_PARAMS = OrderedDict(tmp_k=20.0, tmp_indx_t0=4.0)
TODAY = 13.8


@jjit
def _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, u_ymin, u_dy):
    """
    """
    early_index, late_index = _get_bounded_params(u_ymin, u_dy)
    return _rolling_plaw(logt, logtmp, logmp, x0, k, early_index, late_index)


@jjit
def _rolling_plaw(logt, logtmp, logmp, x0, k, ymin, ymax):
    p = _sigmoid(logt, x0, k, ymin, ymax)
    log_mah = p * (logt - logtmp) + logmp
    return log_mah


@jjit
def _get_bounded_params(u_early_index, u_dy):
    min_early_index, max_early_index = BOUNDS["mah_early_index"]
    early_index = _sigmoid(u_early_index, 0, 1, min_early_index, max_early_index)
    delta_index_max = early_index - min_early_index
    delta_index = _sigmoid(u_dy, 0, 1, 0, delta_index_max)
    late_index = early_index - delta_index
    return early_index, late_index


@jjit
def _get_unbounded_params(early_index, late_index):
    u_early_index = _inverse_sigmoid(early_index, 0, 1, *BOUNDS["mah_early_index"])
    min_early_index = BOUNDS["mah_early_index"][0]
    delta_index_max = early_index - min_early_index
    delta_index = early_index - late_index
    u_dy = _inverse_sigmoid(delta_index, 0, 1, 0, delta_index_max)
    return u_early_index, u_dy


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


def _calc_mah(time, logtmp, logmp, x0, k, u_ymin, u_dy):
    logt = jnp.log10(time)
    return 10 ** _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, u_ymin, u_dy)


@jjit
def _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, u_ymin, u_dy):
    log_mah = _rolling_plaw_bounded(logt, logtmp, logmp, x0, k, u_ymin, u_dy)
    return jnp.where(log_mah > logmp, logmp, log_mah)


@jjit
def _calc_clipped_mah(t, logtmp, logmp, x0, k, u_ymin, u_dy):
    logt = jnp.log10(t)
    return 10 ** _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, u_ymin, u_dy)


_calc_dmhdt = jjit(jvmap(grad(_calc_mah, argnums=0), in_axes=(0, *[None] * 6)))
_calc_clipped_dmhdt = jjit(
    jvmap(grad(_calc_clipped_mah, argnums=0), in_axes=(0, *[None] * 6))
)


@jjit
def _get_clipped_sigmah_kern(logt, logtmp, k, logmp, x0, u_ymin, u_dy):
    log_mah = _calc_clipped_log_mah(logt, logtmp, logmp, x0, k, u_ymin, u_dy)
    dmhdt = _calc_clipped_dmhdt(10 ** logt, logtmp, logmp, x0, k, u_ymin, u_dy)
    return 10 ** log_mah, dmhdt / 1e9


@jjit
def _get_sigmah_halopop_kern(logt, logtmp, k, logmp, x0, early_index, late_index):
    u_early_index, u_dy = _get_unbounded_params(early_index, late_index)
    return _get_clipped_sigmah_kern(logt, logtmp, k, logmp, x0, u_early_index, u_dy)


@jjit
def _get_avg_mah_integrand_kern(logt, logtmp, k, logmp, x0, log10_early_index, u_dy):
    early_index = 10 ** log10_early_index
    u_early_index = _inverse_sigmoid(early_index, 0, 1, *BOUNDS["mah_early_index"])
    return _get_clipped_sigmah_kern(logt, logtmp, k, logmp, x0, u_early_index, u_dy)


_f0 = jvmap(_get_avg_mah_integrand_kern, in_axes=(*[None] * 3, 0, *[None] * 3))
_f1 = jvmap(_f0, in_axes=(None, 0, *[None] * 5))
_f2 = jvmap(_f1, in_axes=(*[None] * 5, 0, None))
_get_avg_mah_early_tmp_integrand = jjit(jvmap(_f2, in_axes=(*[None] * 5, None, 0)))


_g0 = jvmap(_get_avg_mah_integrand_kern, in_axes=(*[None] * 3, 0, *[None] * 3))
_g1 = jvmap(_g0, in_axes=(*[None] * 5, 0, None))
_get_avg_mah_tmp_today_integrand = jjit(jvmap(_g1, in_axes=(*[None] * 5, None, 0)))


@jjit
def _mean_early_index(lgmp, early_x0, early_k, early_ylo, early_yhi):
    return _sigmoid(lgmp, early_x0, early_k, early_ylo, early_yhi)


@jjit
def _mean_late_index(lgmp, late_x0, late_k, late_ylo, late_yhi):
    return _sigmoid(lgmp, late_x0, late_k, late_ylo, late_yhi)


@jjit
def _early_index_u_dy_covariance(log_mah_cov_det=-1.0):
    mah_cov_det = 10 ** log_mah_cov_det
    cov = mah_cov_det * jnp.array(((1 / 4, 1), (1, 8))).astype("f4")
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
        p = [MEAN_MAH_PARAMS[key] for key in MEAN_MAH_PARAMS.keys() if "early" in key]
        early = _mean_early_index(logmp, *p)
    if late is None:
        p = [MEAN_MAH_PARAMS[key] for key in MEAN_MAH_PARAMS.keys() if "late" in key]
        late = _mean_late_index(logmp, *p)

    logmp, logtmp, x0, k, early, late = get_1d_arrays(logmp, logtmp, x0, k, early, late)
    return logt, logmp, logtmp, x0, k, early, late


def _g0(a, b, mu, cov):
    x = jnp.array((a, b)).astype("f4")
    return jnorm.pdf(x, mu, cov)


_g1 = jvmap(_g0, in_axes=(None, None, 0, None))
_g2 = jvmap(_g1, in_axes=(0, None, None, None))
_g3 = jvmap(_g2, in_axes=(None, 0, None, None))


@jjit
def _get_index_weights_kern(log10_early_index, u_dy, mu, cov):
    _pdf = _g3(log10_early_index, u_dy, mu, cov)
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
    early_index_x0,
    early_index_k,
    early_index_ylo,
    early_index_yhi,
    late_index_x0,
    late_index_k,
    late_index_ylo,
    late_index_yhi,
    log_mah_cov_det,
):
    mean_early = _mean_early_index(
        logmparr, early_index_x0, early_index_k, early_index_ylo, early_index_yhi
    )
    mean_late = _mean_late_index(
        logmparr, late_index_x0, late_index_k, late_index_ylo, late_index_yhi
    )
    log10_mean_early = jnp.log10(mean_early)
    mean_u_dy = _get_unbounded_params(mean_early, mean_late)[1]
    X = jnp.array((log10_mean_early, mean_u_dy)).T
    cov = _early_index_u_dy_covariance(log_mah_cov_det)
    mean_index_weights = _get_index_weights_kern(log10_early_arr, u_dy_arr, X, cov)
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
def _log_det_cov_vs_logmp(logmp, log_det_cov_dwarfs, log_det_cov_clusters):
    return _sigmoid(logmp, 13, 1, log_det_cov_dwarfs, log_det_cov_clusters)


@jjit
def _calc_avg_history_early_tmp_halos(
    logt,
    logtmparr,
    k,
    logmparr,
    x0,
    log10_early_arr,
    u_dy_arr,
    early_index_x0,
    early_index_k,
    early_index_ylo,
    early_index_yhi,
    late_index_x0,
    late_index_k,
    late_index_ylo,
    late_index_yhi,
    log_mah_cov_det,
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
        early_index_x0,
        early_index_k,
        early_index_ylo,
        early_index_yhi,
        late_index_x0,
        late_index_k,
        late_index_ylo,
        late_index_yhi,
        log_mah_cov_det,
    )

    tmp_weights = get_tmp_weights_kern(
        logmparr, 10 ** logtmparr, today, tmp_k, tmp_indx_t0
    )
    _w0 = index_weights.reshape((n_late, n_early, 1, n_mass, 1))
    _w1 = tmp_weights.reshape((1, 1, n_tmp, n_mass, 1))
    W = _w0 * _w1
    avg_mah = jnp.sum(W * mah_integrand, axis=(0, 1, 2))
    avg_dmhdt = jnp.sum(W * dmhdt_integrand, axis=(0, 1, 2))
    return avg_mah, avg_dmhdt


@jjit
def _calc_avg_history_tmp_today_halos(
    logt,
    k,
    logmparr,
    x0,
    log10_early_arr,
    u_dy_arr,
    early_index_x0,
    early_index_k,
    early_index_ylo,
    early_index_yhi,
    late_index_x0,
    late_index_k,
    late_index_ylo,
    late_index_yhi,
    log_mah_cov_det,
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
        early_index_x0,
        early_index_k,
        early_index_ylo,
        early_index_yhi,
        late_index_x0,
        late_index_k,
        late_index_ylo,
        late_index_yhi,
        log_mah_cov_det,
    )

    W = index_weights.reshape((n_late, n_early, n_mass, 1))
    avg_mah = jnp.sum(W * mah_integrand, axis=(0, 1))
    avg_dmhdt = jnp.sum(W * dmhdt_integrand, axis=(0, 1))
    return avg_mah, avg_dmhdt


@jjit
def _calc_avg_halo_history(
    logt,
    logtmparr,
    k,
    logmparr,
    x0,
    log10_early_arr,
    u_dy_arr,
    early_index_x0,
    early_index_k,
    early_index_ylo,
    early_index_yhi,
    late_index_x0,
    late_index_k,
    late_index_ylo,
    late_index_yhi,
    log_mah_cov_det,
    tmp_k,
    tmp_indx_t0,
    today,
):
    mah_early, dmhdt_early = _calc_avg_history_early_tmp_halos(
        logt,
        logtmparr,
        k,
        logmparr,
        x0,
        log10_early_arr,
        u_dy_arr,
        early_index_x0,
        early_index_k,
        early_index_ylo,
        early_index_yhi,
        late_index_x0,
        late_index_k,
        late_index_ylo,
        late_index_yhi,
        log_mah_cov_det,
        tmp_k,
        tmp_indx_t0,
        today,
    )

    mah_t0, dmhdt_t0 = _calc_avg_history_tmp_today_halos(
        logt,
        k,
        logmparr,
        x0,
        log10_early_arr,
        u_dy_arr,
        early_index_x0,
        early_index_k,
        early_index_ylo,
        early_index_yhi,
        late_index_x0,
        late_index_k,
        late_index_ylo,
        late_index_yhi,
        log_mah_cov_det,
        tmp_k,
        tmp_indx_t0,
        today,
    )
    n_mass = mah_t0.shape[0]

    f_early = _frac_early_mpeak(logmparr).reshape((n_mass, 1))
    avg_mah = f_early * mah_early + (1 - f_early) * mah_t0
    avg_dmhdt = f_early * dmhdt_early + (1 - f_early) * dmhdt_t0
    return avg_mah, avg_dmhdt


@jjit
def calc_avg_halo_history(params, data):
    args = (*data[0:7], *params, *data[7:])
    return _calc_avg_halo_history(*args)


@jjit
def _mse(pred, target):
    err = (pred - target) / target
    return jnp.sum(err * err)


_mse_bundle = jvmap(_mse, in_axes=(0, 0))


@jjit
def mse_bundle(preds, targets):
    return jnp.sum(_mse_bundle(preds, targets))


@jjit
def _avg_halo_history_loss(params, loss_data):
    target_halo_mahs = loss_data[-1]
    pred_data = loss_data[0:-1]
    avg_mah, avg_dmhdt = calc_avg_halo_history(params, pred_data)
    return mse_bundle(avg_mah, target_halo_mahs)
