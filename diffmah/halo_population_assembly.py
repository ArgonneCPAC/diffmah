"""Calculate differentiable average halo histories."""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax.scipy.stats import multivariate_normal as jnorm
from .individual_halo_assembly import _calc_halo_history, DEFAULT_MAH_PARAMS
from .mah_pop_param_model import frac_late_forming
from .mah_pop_param_model import _get_cov_early, _get_cov_late
from .mah_pop_param_model import _get_mean_mah_params_early, _get_mean_mah_params_late

from .mah_pop_param_model import FRAC_LATE_FORMING_PARAMS
from .mah_pop_param_model import MEAN_PARAMS_EARLY, MEAN_PARAMS_LATE
from .mah_pop_param_model import COV_PARAMS_EARLY, COV_PARAMS_LATE

TODAY = 13.8
LGT0 = jnp.log10(TODAY)


_g0 = vmap(_calc_halo_history, in_axes=(None, None, 0, *[None] * 4))
_g1 = vmap(_g0, in_axes=(*[None] * 5, 0, None))
_g2 = vmap(_g1, in_axes=(*[None] * 5, None, 0))
_halo_history_integrand = jjit(vmap(_g2, in_axes=(*[None] * 3, 0, *[None] * 3)))


def _multivariate_normal_pdf_kernel(lge, lgl, x0, mu, cov):
    X = jnp.array((lge, lgl, x0)).astype("f4")
    return jnorm.pdf(X, mu, cov)


_g0 = vmap(_multivariate_normal_pdf_kernel, in_axes=(None, None, None, 0, 0))
_g1 = vmap(_g0, in_axes=(0, None, None, None, None))
_g2 = vmap(_g1, in_axes=(None, 0, None, None, None))
_get_pdf_weights_kern = vmap(_g2, in_axes=(None, None, 0, None, None))


@jjit
def _get_mah_weights(lge_arr, lgl_arr, x0_arr, mu_arr, cov_arr):
    _pdf = _get_pdf_weights_kern(lge_arr, lgl_arr, x0_arr, mu_arr, cov_arr)
    n_halos = _pdf.shape[-1]
    _norm = jnp.sum(_pdf, axis=(0, 1, 2))
    _norm = jnp.where(_norm == 0, 1.0, _norm)
    pdf = _pdf / _norm.reshape((1, 1, 1, n_halos))
    return pdf


@jjit
def _get_halo_mahs(
    logt,
    logmp_arr,
    lge_arr,
    lgl_arr,
    x0_arr,
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    dmhdt, log_mah = _halo_history_integrand(
        logt, logtmp, logmp_arr, x0_arr, k, 10 ** lge_arr, 10 ** lgl_arr
    )
    return dmhdt, log_mah


@jjit
def _get_average_halo_histories(
    logt,
    logmp_arr,
    lge_arr,
    lgl_arr,
    x0_arr,
    frac_late_forming_lo=FRAC_LATE_FORMING_PARAMS["frac_late_forming_lo"],
    frac_late_forming_hi=FRAC_LATE_FORMING_PARAMS["frac_late_forming_hi"],
    lge_early_lo=MEAN_PARAMS_EARLY["lge_early_lo"],
    lge_early_hi=MEAN_PARAMS_EARLY["lge_early_hi"],
    lgl_early_lo=MEAN_PARAMS_EARLY["lgl_early_lo"],
    lgl_early_hi=MEAN_PARAMS_EARLY["lgl_early_hi"],
    x0_early=MEAN_PARAMS_EARLY["x0_early"],
    lge_late_lo=MEAN_PARAMS_LATE["lge_late_lo"],
    lge_late_hi=MEAN_PARAMS_LATE["lge_late_hi"],
    lgl_late_lo=MEAN_PARAMS_LATE["lgl_late_lo"],
    lgl_late_hi=MEAN_PARAMS_LATE["lgl_late_hi"],
    x0_late=MEAN_PARAMS_LATE["x0_late"],
    log_cho_lge_lge_early_lo=COV_PARAMS_EARLY["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=COV_PARAMS_EARLY["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=COV_PARAMS_EARLY["log_cho_x0_x0_early"],
    cho_lge_lgl_early=COV_PARAMS_EARLY["cho_lge_lgl_early"],
    cho_lge_x0_early=COV_PARAMS_EARLY["cho_lge_x0_early"],
    cho_lgl_x0_early=COV_PARAMS_EARLY["cho_lgl_x0_early"],
    log_cho_lge_lge_late_lo=COV_PARAMS_LATE["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=COV_PARAMS_LATE["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=COV_PARAMS_LATE["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=COV_PARAMS_LATE["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=COV_PARAMS_LATE["log_cho_x0_x0_late"],
    cho_lge_lgl_late=COV_PARAMS_LATE["cho_lge_lgl_late"],
    cho_lge_x0_late=COV_PARAMS_LATE["cho_lge_x0_late"],
    cho_lgl_x0_late=COV_PARAMS_LATE["cho_lgl_x0_late"],
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    dmhdts, log_mahs = _halo_history_integrand(
        logt, logtmp, logmp_arr, x0_arr, k, 10 ** lge_arr, 10 ** lgl_arr
    )
    mahs = 10 ** log_mahs

    lge_early, lgl_early, x0_early = _get_mean_mah_params_early(
        logmp_arr, lge_early_lo, lge_early_hi, lgl_early_lo, lgl_early_hi, x0_early
    )
    means_early = jnp.array((lge_early, lgl_early, x0_early)).T

    lge_late, lgl_late, x0_late = _get_mean_mah_params_late(
        logmp_arr, lge_late_lo, lge_late_hi, lgl_late_lo, lgl_late_hi, x0_late
    )
    means_late = jnp.array((lge_late, lgl_late, x0_late)).T

    covs_late = _get_cov_late(
        logmp_arr,
        log_cho_lge_lge_late_lo,
        log_cho_lge_lge_late_hi,
        log_cho_lgl_lgl_late_lo,
        log_cho_lgl_lgl_late_hi,
        log_cho_x0_x0_late,
        cho_lge_lgl_late,
        cho_lge_x0_late,
        cho_lgl_x0_late,
    )
    covs_early = _get_cov_early(
        logmp_arr,
        log_cho_lge_lge_early_lo,
        log_cho_lge_lge_early_hi,
        log_cho_lgl_lgl_early_lo,
        log_cho_lgl_lgl_early_hi,
        log_cho_x0_x0_early,
        cho_lge_lgl_early,
        cho_lge_x0_early,
        cho_lgl_x0_early,
    )

    weights_early = _get_mah_weights(lge_arr, lgl_arr, x0_arr, means_early, covs_early)
    weights_late = _get_mah_weights(lge_arr, lgl_arr, x0_arr, means_late, covs_late)

    frac_late = frac_late_forming(logmp_arr, frac_late_forming_lo, frac_late_forming_hi)

    n_x0, n_late, n_early, n_halos, n_t = log_mahs.shape
    _wshape = n_x0, n_late, n_early, n_halos, 1
    w_early = ((1 - frac_late) * weights_early).reshape(_wshape)
    w_late = (frac_late * weights_late).reshape(_wshape)

    w_dmhdts = dmhdts * w_early + dmhdts * w_late
    w_mahs = mahs * w_early + mahs * w_late
    mean_dmhdts = jnp.sum(w_dmhdts, axis=(0, 1, 2))
    mean_mahs = jnp.sum(w_mahs, axis=(0, 1, 2))
    mean_log_mahs = jnp.log10(mean_mahs)

    delta_dmhdt_sq = (dmhdts - mean_dmhdts) ** 2
    delta_log_mah_sq = (log_mahs - mean_log_mahs) ** 2

    variance_dmhdt = jnp.sum(
        delta_dmhdt_sq * w_early + delta_dmhdt_sq * w_late, axis=(0, 1, 2)
    )
    variance_log_mah = jnp.sum(
        delta_log_mah_sq * w_early + delta_log_mah_sq * w_late, axis=(0, 1, 2)
    )

    std_dmhdt = jnp.sqrt(variance_dmhdt)
    std_log_mah = jnp.sqrt(variance_log_mah)

    return mean_dmhdts, mean_log_mahs, std_dmhdt, std_log_mah
