"""Calculate differentiable probabilistic history of an individual halo."""
import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax.scipy.stats import multivariate_normal as jnorm
from .individual_halo_assembly import _calc_halo_history, DEFAULT_MAH_PARAMS
from .mah_pop_param_model import frac_late_forming
from .mah_pop_param_model import _get_mean_mah_params_early
from .mah_pop_param_model import _get_mean_mah_params_late
from .mah_pop_param_model import _get_cov_early, _get_cov_late
from .mah_pop_param_model import DEFAULT_MAH_PDF_PARAMS

TODAY = 13.8
LGT0 = jnp.log10(TODAY)
CLIP = -10.0

LGE_ARR = np.linspace(-1, 1.5, 25)
LGL_ARR = np.linspace(-1.5, 1.5, 25)
X0_ARR = np.linspace(-1.5, 1.5, 25)

_g1 = vmap(_calc_halo_history, in_axes=(*[None] * 5, 0, None))
_g2 = vmap(_g1, in_axes=(*[None] * 5, None, 0))
_halo_history_integrand = jjit(vmap(_g2, in_axes=(*[None] * 3, 0, *[None] * 3)))


def _multivariate_normal_pdf_kernel(lge, lgl, x0, mu, cov):
    X = jnp.array((lge, lgl, x0)).astype("f4")
    return jnorm.pdf(X, mu, cov)


_g1 = vmap(_multivariate_normal_pdf_kernel, in_axes=(0, None, None, None, None))
_g2 = vmap(_g1, in_axes=(None, 0, None, None, None))
_get_pdf_weights_kern = vmap(_g2, in_axes=(None, None, 0, None, None))


@jjit
def _get_mah_weights(lge_arr, lgl_arr, x0_arr, mu, cov):
    _pdf = _get_pdf_weights_kern(lge_arr, lgl_arr, x0_arr, mu, cov)
    return _pdf / jnp.sum(_pdf, axis=(0, 1, 2))


@jjit
def _get_bimodal_halo_history_kern(
    logt,
    logmp,
    lge_arr,
    lgl_arr,
    x0_arr,
    frac_late,
    mu_early,
    mu_late,
    cov_early,
    cov_late,
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    dmhdts, log_mahs = _halo_history_integrand(
        logt, logtmp, logmp, x0_arr, k, 10 ** lge_arr, 10 ** lgl_arr
    )
    mahs = 10 ** log_mahs

    weights_early = _get_mah_weights(lge_arr, lgl_arr, x0_arr, mu_early, cov_early)
    weights_late = _get_mah_weights(lge_arr, lgl_arr, x0_arr, mu_late, cov_late)

    n_x0, n_late, n_early, n_t = mahs.shape
    weights_early = weights_early.reshape((n_x0, n_late, n_early, 1))
    weights_late = weights_late.reshape((n_x0, n_late, n_early, 1))

    w = frac_late * weights_late + (1 - frac_late) * weights_early

    mean_dmhdt = jnp.sum(dmhdts * w, axis=(0, 1, 2))
    mean_mah = jnp.sum(mahs * w, axis=(0, 1, 2))
    mean_log_mah = jnp.sum(log_mahs * w, axis=(0, 1, 2))

    delta_dmhdt_sq = (dmhdts - mean_dmhdt) ** 2
    delta_log_mah_sq = (log_mahs - mean_log_mah) ** 2

    variance_dmhdt = jnp.sum(delta_dmhdt_sq * w, axis=(0, 1, 2))
    variance_log_mah = jnp.sum(delta_log_mah_sq * w, axis=(0, 1, 2))

    return mean_dmhdt, mean_mah, mean_log_mah, variance_dmhdt, variance_log_mah


_a = (None, 0, None, None, None, *[0] * 5)
_multimass_bimodal_halo_history_kern = jjit(
    vmap(_get_bimodal_halo_history_kern, in_axes=_a)
)


@jjit
def _get_mah_means_and_covs(
    logmp_arr,
    frac_late_forming_lo=DEFAULT_MAH_PDF_PARAMS["frac_late_forming_lo"],
    frac_late_forming_hi=DEFAULT_MAH_PDF_PARAMS["frac_late_forming_hi"],
    lge_early_lo=DEFAULT_MAH_PDF_PARAMS["lge_early_lo"],
    lge_early_hi=DEFAULT_MAH_PDF_PARAMS["lge_early_hi"],
    lgl_early_lo=DEFAULT_MAH_PDF_PARAMS["lgl_early_lo"],
    lgl_early_hi=DEFAULT_MAH_PDF_PARAMS["lgl_early_hi"],
    x0_early_lo=DEFAULT_MAH_PDF_PARAMS["x0_early_lo"],
    x0_early_hi=DEFAULT_MAH_PDF_PARAMS["x0_early_hi"],
    log_cho_lge_lge_early_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=DEFAULT_MAH_PDF_PARAMS["log_cho_x0_x0_early"],
    cho_lge_lgl_early=DEFAULT_MAH_PDF_PARAMS["cho_lge_lgl_early"],
    cho_lge_x0_early=DEFAULT_MAH_PDF_PARAMS["cho_lge_x0_early"],
    cho_lgl_x0_early=DEFAULT_MAH_PDF_PARAMS["cho_lgl_x0_early"],
    lge_late_lo=DEFAULT_MAH_PDF_PARAMS["lge_late_lo"],
    lge_late_hi=DEFAULT_MAH_PDF_PARAMS["lge_late_hi"],
    lgl_late_lo=DEFAULT_MAH_PDF_PARAMS["lgl_late_lo"],
    lgl_late_hi=DEFAULT_MAH_PDF_PARAMS["lgl_late_hi"],
    x0_late_lo=DEFAULT_MAH_PDF_PARAMS["x0_late_lo"],
    x0_late_hi=DEFAULT_MAH_PDF_PARAMS["x0_late_hi"],
    log_cho_lge_lge_late_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=DEFAULT_MAH_PDF_PARAMS["log_cho_x0_x0_late"],
    cho_lge_lgl_late=DEFAULT_MAH_PDF_PARAMS["cho_lge_lgl_late"],
    cho_lge_x0_late=DEFAULT_MAH_PDF_PARAMS["cho_lge_x0_late"],
    cho_lgl_x0_late=DEFAULT_MAH_PDF_PARAMS["cho_lgl_x0_late"],
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    frac_late = frac_late_forming(logmp_arr, frac_late_forming_lo, frac_late_forming_hi)
    lge_early, lgl_early, x0_early = _get_mean_mah_params_early(
        logmp_arr,
        lge_early_lo,
        lge_early_hi,
        lgl_early_lo,
        lgl_early_hi,
        x0_early_lo,
        x0_early_hi,
    )
    means_early = jnp.array((lge_early, lgl_early, x0_early)).T
    lge_late, lgl_late, x0_late = _get_mean_mah_params_late(
        logmp_arr,
        lge_late_lo,
        lge_late_hi,
        lgl_late_lo,
        lgl_late_hi,
        x0_late_lo,
        x0_late_hi,
    )
    means_late = jnp.array((lge_late, lgl_late, x0_late)).T

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

    return frac_late, means_early, covs_early, means_late, covs_late


@jjit
def _get_bimodal_halo_history(
    logt,
    logmp_arr,
    lge_arr,
    lgl_arr,
    x0_arr,
    frac_late_forming_lo=DEFAULT_MAH_PDF_PARAMS["frac_late_forming_lo"],
    frac_late_forming_hi=DEFAULT_MAH_PDF_PARAMS["frac_late_forming_hi"],
    lge_early_lo=DEFAULT_MAH_PDF_PARAMS["lge_early_lo"],
    lge_early_hi=DEFAULT_MAH_PDF_PARAMS["lge_early_hi"],
    lgl_early_lo=DEFAULT_MAH_PDF_PARAMS["lgl_early_lo"],
    lgl_early_hi=DEFAULT_MAH_PDF_PARAMS["lgl_early_hi"],
    x0_early_lo=DEFAULT_MAH_PDF_PARAMS["x0_early_lo"],
    x0_early_hi=DEFAULT_MAH_PDF_PARAMS["x0_early_hi"],
    log_cho_lge_lge_early_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=DEFAULT_MAH_PDF_PARAMS["log_cho_x0_x0_early"],
    cho_lge_lgl_early=DEFAULT_MAH_PDF_PARAMS["cho_lge_lgl_early"],
    cho_lge_x0_early=DEFAULT_MAH_PDF_PARAMS["cho_lge_x0_early"],
    cho_lgl_x0_early=DEFAULT_MAH_PDF_PARAMS["cho_lgl_x0_early"],
    lge_late_lo=DEFAULT_MAH_PDF_PARAMS["lge_late_lo"],
    lge_late_hi=DEFAULT_MAH_PDF_PARAMS["lge_late_hi"],
    lgl_late_lo=DEFAULT_MAH_PDF_PARAMS["lgl_late_lo"],
    lgl_late_hi=DEFAULT_MAH_PDF_PARAMS["lgl_late_hi"],
    x0_late_lo=DEFAULT_MAH_PDF_PARAMS["x0_late_lo"],
    x0_late_hi=DEFAULT_MAH_PDF_PARAMS["x0_late_hi"],
    log_cho_lge_lge_late_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=DEFAULT_MAH_PDF_PARAMS["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=DEFAULT_MAH_PDF_PARAMS["log_cho_x0_x0_late"],
    cho_lge_lgl_late=DEFAULT_MAH_PDF_PARAMS["cho_lge_lgl_late"],
    cho_lge_x0_late=DEFAULT_MAH_PDF_PARAMS["cho_lge_x0_late"],
    cho_lgl_x0_late=DEFAULT_MAH_PDF_PARAMS["cho_lgl_x0_late"],
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    _res = _get_mah_means_and_covs(
        logmp_arr,
        frac_late_forming_lo,
        frac_late_forming_hi,
        lge_early_lo,
        lge_early_hi,
        lgl_early_lo,
        lgl_early_hi,
        x0_early_lo,
        x0_early_hi,
        log_cho_lge_lge_early_lo,
        log_cho_lge_lge_early_hi,
        log_cho_lgl_lgl_early_lo,
        log_cho_lgl_lgl_early_hi,
        log_cho_x0_x0_early,
        cho_lge_lgl_early,
        cho_lge_x0_early,
        cho_lgl_x0_early,
        lge_late_lo,
        lge_late_hi,
        lgl_late_lo,
        lgl_late_hi,
        x0_late_lo,
        x0_late_hi,
        log_cho_lge_lge_late_lo,
        log_cho_lge_lge_late_hi,
        log_cho_lgl_lgl_late_lo,
        log_cho_lgl_lgl_late_hi,
        log_cho_x0_x0_late,
        cho_lge_lgl_late,
        cho_lge_x0_late,
        cho_lgl_x0_late,
        k,
        logtmp,
    )
    frac_late, means_early, covs_early, means_late, covs_late = _res
    return _multimass_bimodal_halo_history_kern(
        logt,
        logmp_arr,
        lge_arr,
        lgl_arr,
        x0_arr,
        frac_late,
        means_early,
        means_late,
        covs_early,
        covs_late,
    )
