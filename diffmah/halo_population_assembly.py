"""Calculate differentiable probabilistic history of an individual halo."""
import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax.scipy.stats import multivariate_normal as jnorm
from .individual_halo_assembly import _calc_halo_history, _get_early_late
from .individual_halo_assembly import DEFAULT_MAH_PARAMS
from .rockstar_pdf_model import _get_mah_means_and_covs
from .rockstar_pdf_model import DEFAULT_MAH_PDF_PARAMS, LGT0

CLIP = -10.0

UE_ARR = np.linspace(0, 4.5, 30)
UL_ARR = np.linspace(-4.0, 0.5, 30)
LGTC_ARR = np.linspace(-1.5, 2.0, 30)

_g1 = vmap(_calc_halo_history, in_axes=(*[None] * 5, 0, None))
_g2 = vmap(_g1, in_axes=(*[None] * 5, None, 0))
_halo_history_integrand = jjit(vmap(_g2, in_axes=(*[None] * 3, 0, *[None] * 3)))


def _multivariate_normal_pdf_kernel(ue, ul, lgtc, mu, cov):
    X = jnp.array((ue, ul, lgtc)).astype("f4")
    return jnorm.pdf(X, mu, cov)


_g1 = vmap(_multivariate_normal_pdf_kernel, in_axes=(0, None, None, None, None))
_g2 = vmap(_g1, in_axes=(None, 0, None, None, None))
_get_pdf_weights_kern = vmap(_g2, in_axes=(None, None, 0, None, None))


@jjit
def _get_mah_weights(ue_arr, ul_arr, lgtc_arr, mu, cov):
    _pdf = _get_pdf_weights_kern(ue_arr, ul_arr, lgtc_arr, mu, cov)
    return _pdf / jnp.sum(_pdf, axis=(0, 1, 2))


@jjit
def _get_bimodal_halo_history_kern(
    logt,
    logmp,
    ue_arr,
    ul_arr,
    lgtc_arr,
    frac_late,
    mu_early,
    mu_late,
    cov_early,
    cov_late,
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    early_arr, late_arr = _get_early_late(ue_arr, ul_arr)
    dmhdts, log_mahs = _halo_history_integrand(
        logt, logtmp, logmp, lgtc_arr, k, early_arr, late_arr
    )
    mahs = 10 ** log_mahs

    weights_early = _get_mah_weights(ue_arr, ul_arr, lgtc_arr, mu_early, cov_early)
    weights_late = _get_mah_weights(ue_arr, ul_arr, lgtc_arr, mu_late, cov_late)

    n_lgtc, n_late, n_early, n_t = mahs.shape
    weights_early = weights_early.reshape((n_lgtc, n_late, n_early, 1))
    weights_late = weights_late.reshape((n_lgtc, n_late, n_early, 1))

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
def _get_bimodal_halo_history(
    logt,
    logmp_arr,
    ue_arr,
    ul_arr,
    lgtc_arr,
    frac_late_ylo=DEFAULT_MAH_PDF_PARAMS["frac_late_ylo"],
    frac_late_yhi=DEFAULT_MAH_PDF_PARAMS["frac_late_yhi"],
    mean_ue_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ue_early_ylo"],
    mean_ue_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ue_early_yhi"],
    mean_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ul_early_ylo"],
    mean_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ul_early_yhi"],
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
    chol_ue_ue_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_early_ylo"],
    chol_ue_ue_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_early_yhi"],
    chol_ul_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_early_ylo"],
    chol_ul_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_early_yhi"],
    chol_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_early_ylo"],
    chol_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_early_yhi"],
    chol_ue_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_early_ylo"],
    chol_ue_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_early_yhi"],
    chol_ue_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_early_ylo"],
    chol_ue_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_early_yhi"],
    chol_ul_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_early_ylo"],
    chol_ul_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_early_yhi"],
    mean_ue_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ue_late_ylo"],
    mean_ue_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ue_late_yhi"],
    mean_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ul_late_ylo"],
    mean_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ul_late_yhi"],
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
    chol_ue_ue_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_late_ylo"],
    chol_ue_ue_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_late_yhi"],
    chol_ul_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_late_ylo"],
    chol_ul_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_late_yhi"],
    chol_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_late_ylo"],
    chol_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_late_yhi"],
    chol_ue_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_late_ylo"],
    chol_ue_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_late_yhi"],
    chol_ue_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_late_ylo"],
    chol_ue_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_late_yhi"],
    chol_ul_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_late_ylo"],
    chol_ul_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_late_yhi"],
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    _res = _get_mah_means_and_covs(
        logmp_arr,
        frac_late_ylo,
        frac_late_yhi,
        mean_ue_early_ylo,
        mean_ue_early_yhi,
        mean_ul_early_ylo,
        mean_ul_early_yhi,
        mean_lgtc_early_ylo,
        mean_lgtc_early_yhi,
        chol_ue_ue_early_ylo,
        chol_ue_ue_early_yhi,
        chol_ul_ul_early_ylo,
        chol_ul_ul_early_yhi,
        chol_lgtc_lgtc_early_ylo,
        chol_lgtc_lgtc_early_yhi,
        chol_ue_ul_early_ylo,
        chol_ue_ul_early_yhi,
        chol_ue_lgtc_early_ylo,
        chol_ue_lgtc_early_yhi,
        chol_ul_lgtc_early_ylo,
        chol_ul_lgtc_early_yhi,
        mean_ue_late_ylo,
        mean_ue_late_yhi,
        mean_ul_late_ylo,
        mean_ul_late_yhi,
        mean_lgtc_late_ylo,
        mean_lgtc_late_yhi,
        chol_ue_ue_late_ylo,
        chol_ue_ue_late_yhi,
        chol_ul_ul_late_ylo,
        chol_ul_ul_late_yhi,
        chol_lgtc_lgtc_late_ylo,
        chol_lgtc_lgtc_late_yhi,
        chol_ue_ul_late_ylo,
        chol_ue_ul_late_yhi,
        chol_ue_lgtc_late_ylo,
        chol_ue_lgtc_late_yhi,
        chol_ul_lgtc_late_ylo,
        chol_ul_lgtc_late_yhi,
        k,
        logtmp,
    )
    frac_late, means_early, covs_early, means_late, covs_late = _res
    return _multimass_bimodal_halo_history_kern(
        logt,
        logmp_arr,
        ue_arr,
        ul_arr,
        lgtc_arr,
        frac_late,
        means_early,
        means_late,
        covs_early,
        covs_late,
    )
