"""
"""
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from jax.scipy.stats import multivariate_normal as jnorm
from scipy.stats import multivariate_normal as snorm
from .sigmah import _get_clipped_log_mah_kern, _get_bounded_params
from .sigmah import individual_halo_assembly


F_EARLY_PARAMS = OrderedDict(
    f_early_x0=11.7, f_early_k=1, f_early_ylo=0.825, f_early_yhi=0.278
)

MEAN_LGE = OrderedDict(lge_x0=14.79, lge_k=0.84, lge_ylo=0.448, lge_yhi=0.75)
MEAN_U_DY = OrderedDict(u_dy_x0=12.0, u_dy_k=0.85, u_dy_ylo=2.0, u_dy_yhi=-0.18)
MEAN_COV_U = OrderedDict(cov_u_x0=13.0, cov_u_k=0.2, cov_u_ylo=-1.25, cov_u_yhi=-7.75)
MEAN_COV_V = OrderedDict(cov_v_x0=14.25, cov_v_k=2.5, cov_v_ylo=1.2, cov_v_yhi=1.55)
MEAN_COV_Z = OrderedDict(cov_z_x0=12.1, cov_z_k=1.5, cov_z_ylo=-1.35, cov_z_yhi=-2.2)
LGD_BOUNDS = OrderedDict(cov_lgd_x0=-2, cov_lgd_k=0.5, cov_lgd_ylo=-5, cov_lgd_yhi=1)

MEAN_INDEX_PARAMS = deepcopy(MEAN_LGE)
MEAN_INDEX_PARAMS.update(deepcopy(MEAN_U_DY))
COV_INDEX_PARAMS = deepcopy(MEAN_COV_U)
COV_INDEX_PARAMS.update(deepcopy(MEAN_COV_V))
COV_INDEX_PARAMS.update(deepcopy(MEAN_COV_Z))
TMP_PDF_PARAMS = OrderedDict(tmp_k=20.0, tmp_indx_t0=5.54)
TMP_X0_PARAMS = OrderedDict(tmp_x0_x0=13, tmp_x0_k=1.5, tmp_x0_ylo=0.775, tmp_x0_yhi=1)
TMP_FALLOFF_PARAMS = OrderedDict(
    tmp_falloff_x0=12.465, tmp_falloff_k=1.6, tmp_falloff_yhi=13.69
)


@jjit
def _plaw_indices_mean_and_cov(logmp, mean_params, cov_params):
    params = _index_params_vs_logmp(logmp, mean_params, cov_params)
    lge, u_dy, u, v, z = params
    cov = _get_cov_from_u_v_z_arrays(u, v, z)
    return lge, u_dy, cov


@jjit
def _index_params_vs_logmp(logmp, mean_params, cov_params):
    log_early = _mean_lge_vs_logmp(logmp, *mean_params[:4])
    u_dy = _mean_u_dy_vs_logmp(logmp, *mean_params[4:8])
    cov_u = _mean_cov_u_vs_logmp(logmp, *cov_params[:4])
    cov_v = _mean_cov_v_vs_logmp(logmp, *cov_params[4:8])
    cov_z = _mean_cov_z_vs_logmp(logmp, *cov_params[8:12])
    return log_early, u_dy, cov_u, cov_v, cov_z


@jjit
def _mean_lge_vs_logmp(logmp, lge_x0, lge_k, lge_ylo, lge_yhi):
    return _sigmoid(logmp, lge_x0, lge_k, lge_ylo, lge_yhi)


@jjit
def _mean_u_dy_vs_logmp(logmp, u_dy_x0, u_dy_k, u_dy_ylo, u_dy_yhi):
    return _sigmoid(logmp, u_dy_x0, u_dy_k, u_dy_ylo, u_dy_yhi)


@jjit
def _mean_cov_u_vs_logmp(logmp, cov_u_x0, cov_u_k, cov_u_ylo, cov_u_yhi):
    return _sigmoid(logmp, cov_u_x0, cov_u_k, cov_u_ylo, cov_u_yhi)


@jjit
def _mean_cov_v_vs_logmp(logmp, cov_v_x0, cov_v_k, cov_v_ylo, cov_v_yhi):
    return _sigmoid(logmp, cov_v_x0, cov_v_k, cov_v_ylo, cov_v_yhi)


@jjit
def _mean_cov_z_vs_logmp(logmp, cov_z_x0, cov_z_k, cov_z_ylo, cov_z_yhi):
    return _sigmoid(logmp, cov_z_x0, cov_z_k, cov_z_ylo, cov_z_yhi)


@jjit
def _mean_cov_lgd_vs_logmp(logmp, cov_z_x0, cov_z_k, cov_z_ylo, cov_z_yhi):
    cov_z = _mean_cov_z_vs_logmp(logmp, cov_z_x0, cov_z_k, cov_z_ylo, cov_z_yhi)
    return _get_cov_lgd_from_cov_z(cov_z)


@jjit
def _get_cov_lgd_from_cov_z(cov_z):
    x0, k, ylo, yhi = (
        LGD_BOUNDS["cov_lgd_x0"],
        LGD_BOUNDS["cov_lgd_k"],
        LGD_BOUNDS["cov_lgd_ylo"],
        LGD_BOUNDS["cov_lgd_yhi"],
    )
    cov_lgd = _sigmoid(cov_z, x0, k, ylo, yhi)
    return cov_lgd


@jjit
def _get_cov_z_from_cov_lgd(log10_det):
    x0, k, ylo, yhi = (
        LGD_BOUNDS["cov_lgd_x0"],
        LGD_BOUNDS["cov_lgd_k"],
        LGD_BOUNDS["cov_lgd_ylo"],
        LGD_BOUNDS["cov_lgd_yhi"],
    )
    cov_z = _sigmoid(log10_det, x0, k, ylo, yhi)
    return cov_z


@jjit
def _2d_covariance(u, v, log10_det):
    _x = jnp.sqrt(1.0 + u * u + v * v)
    det = 10.0 ** log10_det
    cov = jnp.sqrt(det) * jnp.array(((_x + u, v), (v, _x - u)))
    return cov


@jjit
def _get_cov_from_u_v_z(u, v, cov_z):
    log10_det = _get_cov_lgd_from_cov_z(cov_z)
    cov = _2d_covariance(u, v, log10_det)
    return cov


_get_cov_from_u_v_z_arrays = jjit(jvmap(_get_cov_from_u_v_z, in_axes=(0, 0, 0)))


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _get_cov_params(cov):
    det = jnp.linalg.det(cov)
    M = cov / jnp.sqrt(det)
    v = M[1, 0]
    log10_det = jnp.log10(det)
    u = 0.5 * (M[0, 0] - M[1, 1])
    return u, v, log10_det


@jjit
def _prob_tmp_early_forming_cens(
    logm0,
    tmp,
    t0,
    tmp_k,
    tmp_indx_t0,
    tmp_x0_x0,
    tmp_x0_k,
    tmp_x0_ylo,
    tmp_x0_yhi,
    tmp_falloff_x0,
    tmp_falloff_k,
    tmp_falloff_yhi,
):
    alpha = _prob_tmp_indx(
        logm0,
        tmp,
        tmp_k,
        tmp_indx_t0,
        tmp_x0_x0,
        tmp_x0_k,
        tmp_x0_ylo,
        tmp_x0_yhi,
        tmp_falloff_x0,
        tmp_falloff_k,
        tmp_falloff_yhi,
    )
    pdf_tmpeak_early = _tmp_pdf_powlaw(tmp, alpha, t0)
    return pdf_tmpeak_early


@jjit
def _prob_tmp_indx(
    logm0,
    tmp,
    tmp_k,
    tmp_indx_t0,
    tmp_x0_x0,
    tmp_x0_k,
    tmp_x0_ylo,
    tmp_x0_yhi,
    tmp_falloff_x0,
    tmp_falloff_k,
    tmp_falloff_yhi,
):
    tmp_x0 = _get_tmp_x0_arr(logm0, tmp_x0_x0, tmp_x0_k, tmp_x0_ylo, tmp_x0_yhi)
    tmp_falloff = _get_tmp_falloff_arr(
        logm0, tmp_falloff_x0, tmp_falloff_k, tmp_indx_t0, tmp_falloff_yhi
    )
    alpha = _sigmoid(jnp.log10(tmp), tmp_x0, tmp_k, tmp_falloff, tmp_indx_t0)
    return alpha


@jjit
def _get_tmp_x0_arr(logm0, tmp_x0_x0, tmp_x0_k, tmp_x0_ylo, tmp_x0_yhi):
    return _sigmoid(logm0, tmp_x0_x0, tmp_x0_k, tmp_x0_ylo, tmp_x0_yhi)


@jjit
def _get_tmp_falloff_arr(
    logm0, tmp_falloff_x0, tmp_falloff_k, tmp_indx_t0, tmp_falloff_yhi
):
    return _sigmoid(logm0, tmp_falloff_x0, tmp_falloff_k, tmp_indx_t0, tmp_falloff_yhi)


@jjit
def _tmp_pdf_powlaw(t, indx, t0):
    x = jnp.where(t > t0, 1, t / t0)
    return (indx / t0) * jnp.power(x, indx - 1)


_h0 = jvmap(_prob_tmp_early_forming_cens, in_axes=(None, 0, *[None] * 10))


@jjit
def _normed_prob_tmp(
    logm0,
    tmparr,
    t0,
    tmp_k,
    tmp_indx_t0,
    tmp_x0_x0,
    tmp_x0_k,
    tmp_x0_ylo,
    tmp_x0_yhi,
    tmp_falloff_x0,
    tmp_falloff_k,
    tmp_falloff_yhi,
):
    _pdf = _h0(
        logm0,
        tmparr,
        t0,
        tmp_k,
        tmp_indx_t0,
        tmp_x0_x0,
        tmp_x0_k,
        tmp_x0_ylo,
        tmp_x0_yhi,
        tmp_falloff_x0,
        tmp_falloff_k,
        tmp_falloff_yhi,
    )
    pdf = _pdf / _pdf.sum()
    return pdf


_get_tmp_weights_kern = jjit(jvmap(_normed_prob_tmp, in_axes=(0, *[None] * 11)))


@jjit
def get_tmp_weights_kern(
    logm0,
    tmparr,
    t0,
    tmp_k,
    tmp_indx_t0,
    tmp_x0_x0,
    tmp_x0_k,
    tmp_x0_ylo,
    tmp_x0_yhi,
    tmp_falloff_x0,
    tmp_falloff_k,
    tmp_falloff_yhi,
):
    return _get_tmp_weights_kern(
        logm0,
        tmparr,
        t0,
        tmp_k,
        tmp_indx_t0,
        tmp_x0_x0,
        tmp_x0_k,
        tmp_x0_ylo,
        tmp_x0_yhi,
        tmp_falloff_x0,
        tmp_falloff_k,
        tmp_falloff_yhi,
    ).T


@jjit
def _frac_early_mpeak(logmp, f_early_x0, f_early_k, f_early_ylo, f_early_yhi):
    """Fraction of Rockstar centrals with t_Mpeak < today."""
    return _sigmoid(logmp, f_early_x0, f_early_k, f_early_ylo, f_early_yhi)


def _multivariate_normal_pdf_kernel(a, b, mu, cov):
    x = jnp.array((a, b)).astype("f4")
    return jnorm.pdf(x, mu, cov)


_g1 = jvmap(_multivariate_normal_pdf_kernel, in_axes=(None, None, 0, 0))
_g2 = jvmap(_g1, in_axes=(0, None, None, None))
_get_index_weights_kern = jvmap(_g2, in_axes=(None, 0, None, None))


@jjit
def _get_index_weights(lge_arr, u_dy_arr, mu_arr, cov_arr):
    _pdf = _get_index_weights_kern(lge_arr, u_dy_arr, mu_arr, cov_arr)
    n_mass = _pdf.shape[2]
    _norm = jnp.sum(_pdf, axis=(0, 1))
    _norm = jnp.where(_norm == 0, 1.0, _norm)
    pdf = _pdf / _norm.reshape((1, 1, n_mass))
    return pdf


@jjit
def _get_mean_index_weights(logmp_arr, lge_arr, u_dy_arr, mean_params, cov_params):
    _res = _plaw_indices_mean_and_cov(logmp_arr, mean_params, cov_params)
    mean_lge_arr, mean_u_dy_arr, cov_arr = _res
    X = jnp.array((mean_lge_arr, mean_u_dy_arr)).T
    index_weights = _get_index_weights(lge_arr, u_dy_arr, X, cov_arr)
    return index_weights


_f0 = jvmap(_get_clipped_log_mah_kern, in_axes=(*[None] * 3, 0, *[None] * 3))
_f1 = jvmap(_f0, in_axes=(None, 0, *[None] * 5))
_f2 = jvmap(_f1, in_axes=(*[None] * 5, 0, None))
_get_avg_mah_early_tmp_integrand = jjit(jvmap(_f2, in_axes=(*[None] * 5, None, 0)))


_g0 = jvmap(_get_clipped_log_mah_kern, in_axes=(*[None] * 3, 0, *[None] * 3))
_g1 = jvmap(_g0, in_axes=(*[None] * 5, 0, None))
_get_avg_mah_tmp_today_integrand = jjit(jvmap(_g1, in_axes=(*[None] * 5, None, 0)))


@jjit
def _weighted_history_early_tmp_halos(
    mean_index_params,
    cov_index_params,
    tmp_pdf_params,
    tmp_x0_params,
    tmp_falloff_params,
    mah_x0,
    mah_k,
    logt,
    logmparr,
    logtmparr,
    lge_arr,
    u_dy_arr,
    today,
):
    log_mah_integrand, dmhdt_integrand = _get_avg_mah_early_tmp_integrand(
        logt, logtmparr, mah_k, logmparr, mah_x0, lge_arr, u_dy_arr
    )
    mah_integrand = 10 ** log_mah_integrand
    n_late, n_early, n_tmp, n_mass, n_times = mah_integrand.shape

    index_weights = _get_mean_index_weights(
        logmparr, lge_arr, u_dy_arr, mean_index_params, cov_index_params
    )

    tmp_weights = get_tmp_weights_kern(
        logmparr,
        10 ** logtmparr,
        today,
        *tmp_pdf_params,
        *tmp_x0_params,
        *tmp_falloff_params
    )
    _w0 = index_weights.reshape((n_late, n_early, 1, n_mass, 1))
    _w1 = tmp_weights.reshape((1, 1, n_tmp, n_mass, 1))
    weights = _w0 * _w1

    return mah_integrand, dmhdt_integrand, weights


@jjit
def _weighted_history_tmp_today_halos(
    mean_index_params,
    cov_index_params,
    mah_x0,
    mah_k,
    logt,
    logmparr,
    lge_arr,
    u_dy_arr,
    today,
):
    log_mah_integrand, dmhdt_integrand = _get_avg_mah_tmp_today_integrand(
        logt, jnp.log10(today), mah_k, logmparr, mah_x0, lge_arr, u_dy_arr
    )
    mah_integrand = 10 ** log_mah_integrand
    n_late, n_early, n_mass, n_times = mah_integrand.shape

    index_weights = _get_mean_index_weights(
        logmparr, lge_arr, u_dy_arr, mean_index_params, cov_index_params,
    )

    weights = index_weights.reshape((n_late, n_early, n_mass, 1))
    return mah_integrand, dmhdt_integrand, weights


def _integrate_halos(histories, weights, axis):
    return jnp.sum(weights * histories, axis=axis)


@jjit
def _calc_avg_halo_history(
    mean_index_params,
    cov_index_params,
    tmp_pdf_params,
    tmp_x0_params,
    tmp_falloff_params,
    f_early_params,
    mah_x0,
    mah_k,
    logt,
    logmparr,
    logtmparr,
    lge_arr,
    u_dy_arr,
    today,
):
    _res_early = _weighted_history_early_tmp_halos(
        mean_index_params,
        cov_index_params,
        tmp_pdf_params,
        tmp_x0_params,
        tmp_falloff_params,
        mah_x0,
        mah_k,
        logt,
        logmparr,
        logtmparr,
        lge_arr,
        u_dy_arr,
        today,
    )
    mah_int_early, dmhdt_int_early, w_early = _res_early

    _res_t0 = _weighted_history_tmp_today_halos(
        mean_index_params,
        cov_index_params,
        mah_x0,
        mah_k,
        logt,
        logmparr,
        lge_arr,
        u_dy_arr,
        today,
    )
    mah_int_t0, dmhdt_int_t0, w_t0 = _res_t0

    n_late, n_early, n_tmp, n_mass, n_times = mah_int_early.shape

    mah_int_t0_rs = mah_int_t0.reshape((n_late, n_early, 1, n_mass, n_times))
    dmhdt_int_t0_rs = dmhdt_int_t0.reshape((n_late, n_early, 1, n_mass, n_times))
    w_t0_rs = w_t0.reshape((n_late, n_early, 1, n_mass, 1)) / n_tmp

    F_e = _frac_early_mpeak(logmparr, *f_early_params).reshape((n_mass, 1))

    mah_int_early_term = F_e * w_early * mah_int_early
    mah_int_late_term = (1 - F_e) * w_t0_rs * mah_int_t0_rs
    mah_int = mah_int_early_term + mah_int_late_term

    dmhdt_int_early_term = F_e * w_early * dmhdt_int_early
    dmhdt_int_late_term = (1 - F_e) * w_t0_rs * dmhdt_int_t0_rs
    dmhdt_int = dmhdt_int_early_term + dmhdt_int_late_term

    avg_mah = jnp.sum(mah_int, axis=(0, 1, 2))
    avg_dmhdt = jnp.sum(dmhdt_int, axis=(0, 1, 2))

    d_dmhdt_early_sq = (dmhdt_int_early - avg_dmhdt) ** 2
    d_dmhdt_late_sq = (dmhdt_int_t0_rs - avg_dmhdt) ** 2

    var_term1 = F_e * w_early * d_dmhdt_early_sq
    var_term2 = (1 - F_e) * w_t0_rs * d_dmhdt_late_sq
    var_dmhdt = jnp.sum(var_term1 + var_term2, axis=(0, 1, 2))
    std_dmhdt = jnp.sqrt(var_dmhdt)

    return avg_mah, avg_dmhdt, std_dmhdt


@jjit
def calc_avg_halo_history(params, pred_data):
    return _calc_avg_halo_history(params, *pred_data)


@jjit
def _calc_avg_halo_history_masked(
    mean_index_params,
    cov_index_params,
    tmp_pdf_params,
    tmp_x0_params,
    tmp_falloff_params,
    f_early_params,
    mah_x0,
    mah_k,
    logt,
    logmparr,
    logtmparr,
    lge_arr,
    u_dy_arr,
    today,
    mp_mins,
):
    _res_early = _weighted_history_early_tmp_halos(
        mean_index_params,
        cov_index_params,
        tmp_pdf_params,
        tmp_x0_params,
        tmp_falloff_params,
        mah_x0,
        mah_k,
        logt,
        logmparr,
        logtmparr,
        lge_arr,
        u_dy_arr,
        today,
    )
    mah_int_early, dmhdt_int_early, w_early = _res_early

    _res_t0 = _weighted_history_tmp_today_halos(
        mean_index_params,
        cov_index_params,
        mah_x0,
        mah_k,
        logt,
        logmparr,
        lge_arr,
        u_dy_arr,
        today,
    )
    mah_int_t0, dmhdt_int_t0, w_t0 = _res_t0

    n_late, n_early, n_tmp, n_mass, n_t = mah_int_early.shape

    mp_mins_early_rs = mp_mins.reshape((1, 1, 1, n_mass, 1))
    m_min_msk_early = (mah_int_early >= mp_mins_early_rs).astype("f4")
    mp_mins_t0_rs = mp_mins.reshape((1, 1, n_mass, 1))
    m_min_msk_t0 = (mah_int_t0 >= mp_mins_t0_rs).astype("f4")

    mah_int_t0_rs = mah_int_t0.reshape((n_late, n_early, 1, n_mass, n_t))
    dmhdt_int_t0_rs = dmhdt_int_t0.reshape((n_late, n_early, 1, n_mass, n_t))
    w_t0_rs = w_t0.reshape((n_late, n_early, 1, n_mass, 1))
    m_min_msk_t0_rs = m_min_msk_t0.reshape((n_late, n_early, 1, n_mass, n_t))

    F_e = _frac_early_mpeak(logmparr, *f_early_params).reshape((n_mass, 1))

    W_early_norm = jnp.sum(w_early * m_min_msk_early, axis=(0, 1, 2))
    W_t0_norm = jnp.sum(w_t0_rs * m_min_msk_t0_rs, axis=(0, 1, 2))
    W_early = w_early * m_min_msk_early / W_early_norm
    W_t0 = w_t0_rs * m_min_msk_t0_rs / W_t0_norm

    mah_int_early_term = F_e * W_early * mah_int_early
    mah_int_t0_term = (1 - F_e) * W_t0 * mah_int_t0_rs
    mah_int = W_early * mah_int_early
    mah_int = mah_int_early_term + mah_int_t0_term / n_tmp

    dmhdt_int_early_term = F_e * W_early * dmhdt_int_early
    dmhdt_int_t0_term = (1 - F_e) * W_t0 * dmhdt_int_t0_rs
    dmhdt_int = dmhdt_int_early_term + dmhdt_int_t0_term / n_tmp

    avg_mah = jnp.sum(mah_int, axis=(0, 1, 2))
    avg_dmhdt = jnp.sum(dmhdt_int, axis=(0, 1, 2))

    d_dmhdt_early_sq = (dmhdt_int_early - avg_dmhdt) ** 2
    d_dmhdt_late_sq = (dmhdt_int_t0_rs - avg_dmhdt) ** 2

    var_early_term = F_e * W_early * d_dmhdt_early_sq
    var_t0_term = (1 - F_e) * W_t0 * d_dmhdt_late_sq
    var_dmhdt = jnp.sum(var_early_term + var_t0_term / n_tmp, axis=(0, 1, 2))
    std_dmhdt = jnp.sqrt(var_dmhdt)

    return avg_mah, avg_dmhdt, std_dmhdt


@jjit
def calc_avg_halo_history_masked(params, pred_data):
    return _calc_avg_halo_history_masked(params, *pred_data)


@jjit
def _mse(pred, target):
    err = (pred - target) / target
    return jnp.mean(err * err)


_mse_bundle = jvmap(_mse, in_axes=(0, 0))


@jjit
def mse_bundle(preds, targets):
    return jnp.sum(_mse_bundle(preds, targets))


@jjit
def _loss_with_a_few_tmp_params_varied(params, loss_data):
    mean_index_params = params[:8]
    f_early_yhi, tmp_indx_t0, tmp_falloff_x0, tmp_falloff_yhi = params[8:]

    default_tmp_falloff = list(TMP_FALLOFF_PARAMS.values())
    tmp_falloff_p = jnp.array((tmp_falloff_x0, default_tmp_falloff[1], tmp_falloff_yhi))

    default_tmp_pdf_p = list(TMP_PDF_PARAMS.values())
    tmp_pdf_p = jnp.array((default_tmp_pdf_p[0], tmp_indx_t0)).astype("f4")

    default_f_early_p = list(F_EARLY_PARAMS.values())
    f_early_p = jnp.array((*default_f_early_p[0:-1], f_early_yhi)).astype("f4")

    target_halo_mahs, target_halo_dmhdts = loss_data[-2:]
    _p = loss_data[0:-2]

    pred_data = (_p[0], tmp_pdf_p, _p[2], tmp_falloff_p, f_early_p, *_p[5:])
    _pred = calc_avg_halo_history(mean_index_params, pred_data)
    avg_mah, avg_dmhdt, avg_smar, var_smar = _pred
    loss_mah = mse_bundle(avg_mah, target_halo_mahs)
    loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
    return loss_mah + loss_dmhdt


@jjit
def _avg_halo_history_loss(params, loss_data):
    target_halo_mahs = loss_data[-1]
    pred_data = loss_data[0:-1]
    avg_mah, avg_dmhdt, std_dmhdt = calc_avg_halo_history(params, pred_data)
    return mse_bundle(avg_mah, target_halo_mahs)


@jjit
def _avg_halo_history_loss2(params, loss_data):
    target_halo_mahs, target_halo_dmhdts = loss_data[-2:]
    pred_data = loss_data[0:-2]
    avg_mah, avg_dmhdt, std_dmhdt = calc_avg_halo_history(params, pred_data)
    loss_mah = mse_bundle(avg_mah, target_halo_mahs)
    loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
    return loss_mah + loss_dmhdt


@jjit
def _avg_halo_history_loss2b(params, loss_data):
    target_halo_mahs, target_halo_dmhdts = loss_data[-2:]
    pred_data = loss_data[0:-2]
    avg_mah, avg_dmhdt, std_dmhdt = calc_avg_halo_history_masked(params, pred_data)
    loss_mah = mse_bundle(avg_mah, target_halo_mahs)
    loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
    return loss_mah + loss_dmhdt


@jjit
def _avg_halo_history_loss2c(params, loss_data):
    mean_index_params = params[0:8]
    cov_index_params = params[8:]
    target_halo_mahs, target_halo_dmhdts = loss_data[-2:]
    pred_data = (cov_index_params, *loss_data[0:-2])
    avg_mah, avg_dmhdt, std_dmhdt = calc_avg_halo_history_masked(
        mean_index_params, pred_data
    )
    loss_mah = mse_bundle(avg_mah, target_halo_mahs)
    loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
    return loss_mah + loss_dmhdt


# @jjit
# def _avg_halo_history_loss2d(params, loss_data):
#     mean_index_params = params[:8]
#     cov_index_params = params[8:20]
#     f_early_yhi, tmp_indx_t0, tmp_falloff_x0, tmp_falloff_yhi = params[20:]
#
#     default_tmp_falloff = list(TMP_FALLOFF_PARAMS.values())
#     tmp_falloff_p = jnp.array((tmp_falloff_x0, default_tmp_falloff[1], tmp_falloff_yhi))
#
#     default_tmp_pdf_p = list(TMP_PDF_PARAMS.values())
#     tmp_pdf_p = jnp.array((default_tmp_pdf_p[0], tmp_indx_t0)).astype("f4")
#
#     default_f_early_p = list(F_EARLY_PARAMS.values())
#     f_early_p = jnp.array((*default_f_early_p[0:-1], f_early_yhi)).astype("f4")
#
#     target_halo_mahs, target_halo_dmhdts = loss_data[-2:]
#     _p = loss_data[0:-2]
#
#     pred_data = (
#         cov_index_params,
#         _p[0],
#         tmp_pdf_p,
#         _p[2],
#         tmp_falloff_p,
#         f_early_p,
#         *_p[5:],
#     )
#     _pred = calc_avg_halo_history_masked(mean_index_params, pred_data)
#     avg_mah, avg_dmhdt, std_dmhdt = _pred
#     loss_mah = mse_bundle(avg_mah, target_halo_mahs)
#     loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
#     return loss_mah + loss_dmhdt


@jjit
def _avg_halo_history_loss3(params, loss_data):
    target_halo_mahs, target_halo_dmhdts, target_std_dmhdt = loss_data[-3:]
    pred_data = loss_data[0:-3]
    avg_mah, avg_dmhdt, std_dmhdt = calc_avg_halo_history_masked(params, pred_data)
    loss_mah = mse_bundle(avg_mah, target_halo_mahs)
    loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
    loss_std = mse_bundle(std_dmhdt, target_std_dmhdt)
    return loss_mah + loss_dmhdt + loss_std


@jjit
def _avg_halo_history_loss3b(params, loss_data):
    mean_index_params = params[0:8]
    cov_index_params = params[8:]
    target_halo_mahs, target_halo_dmhdts, target_std_dmhdt = loss_data[-3:]
    pred_data = (cov_index_params, *loss_data[0:-3])
    avg_mah, avg_dmhdt, std_dmhdt = calc_avg_halo_history_masked(
        mean_index_params, pred_data
    )
    loss_mah = mse_bundle(avg_mah, target_halo_mahs)
    loss_dmhdt = mse_bundle(avg_dmhdt, target_halo_dmhdts)
    loss_std = mse_bundle(std_dmhdt, target_std_dmhdt)
    return loss_mah + loss_dmhdt  # + loss_std


def measure_cov_params(cov):
    assert cov[0, 1] == cov[1, 0]
    det = jnp.linalg.det(cov)
    M = cov / jnp.sqrt(det)
    v = M[1, 0]
    log10_det = jnp.log10(det)
    u = 0.5 * (M[0, 0] - M[1, 1])
    return u, v, log10_det


def generate_halo_histories(logmp, t, n_halos, **kw):
    mean_p = np.array(list(MEAN_INDEX_PARAMS.values()))
    cov_p = np.array(list(COV_INDEX_PARAMS.values()))
    mean_params = _index_params_vs_logmp(logmp, mean_p, cov_p)
    mean_lge, mean_u_dy, cov_u, cov_v, cov_z = mean_params
    cov = _get_cov_from_u_v_z(cov_u, cov_v, cov_z)

    X = np.array((mean_lge, mean_u_dy))
    assembly_pdf = snorm.rvs(X, cov, size=n_halos)
    lge, u_dy = assembly_pdf[:, 0], assembly_pdf[:, 1]
    early, late = _get_bounded_params(lge, u_dy)

    log_mah, dmhdt = individual_halo_assembly(t, logmp, early, late, **kw)
    halopop_data = (X, cov, assembly_pdf, early, late, lge, u_dy, mean_params)
    return log_mah, dmhdt, halopop_data
