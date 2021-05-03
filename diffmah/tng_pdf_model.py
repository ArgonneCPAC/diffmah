"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax import vmap
from .individual_halo_assembly import DEFAULT_MAH_PARAMS

TODAY = 13.8
LGT0 = jnp.log10(TODAY)
K = DEFAULT_MAH_PARAMS["mah_k"]

_LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_MAH_PDF_PARAMS = OrderedDict(
    frac_late_ylo=0.30,
    frac_late_yhi=0.80,
    mean_ue_early_ylo=0.68,
    mean_ue_early_yhi=3.84,
    mean_ul_early_ylo=-0.80,
    mean_ul_early_yhi=-0.79,
    mean_lgtc_early_ylo=-0.61,
    mean_lgtc_early_yhi=1.25,
    chol_ue_ue_early_ylo=0.00,
    chol_ue_ue_early_yhi=-0.24,
    chol_ul_ul_early_ylo=-0.10,
    chol_ul_ul_early_yhi=0.05,
    chol_lgtc_lgtc_early_ylo=-0.30,
    chol_lgtc_lgtc_early_yhi=-1.32,
    chol_ue_ul_early_ylo=-0.75,
    chol_ue_ul_early_yhi=-0.65,
    chol_ue_lgtc_early_ylo=-0.15,
    chol_ue_lgtc_early_yhi=0.00,
    chol_ul_lgtc_early_ylo=-0.04,
    chol_ul_lgtc_early_yhi=-0.25,
    mean_ue_late_ylo=0.30,
    mean_ue_late_yhi=2.80,
    mean_ul_late_ylo=-2.64,
    mean_ul_late_yhi=-1.48,
    mean_lgtc_late_ylo=0.10,
    mean_lgtc_late_yhi=2.15,
    chol_ue_ue_late_ylo=-0.16,
    chol_ue_ue_late_yhi=-0.60,
    chol_ul_ul_late_ylo=-0.10,
    chol_ul_ul_late_yhi=-0.10,
    chol_lgtc_lgtc_late_ylo=-0.15,
    chol_lgtc_lgtc_late_yhi=-0.97,
    chol_ue_ul_late_ylo=-1.00,
    chol_ue_ul_late_yhi=0.65,
    chol_ue_lgtc_late_ylo=0.00,
    chol_ue_lgtc_late_yhi=0.04,
    chol_ul_lgtc_late_ylo=0.60,
    chol_ul_lgtc_late_yhi=0.15,
)


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


def _get_cov_scalar(
    log10_lge_lge,
    log10_lgl_lgl,
    log10_lgtc_lgtc,
    lge_lgl,
    lge_lgtc,
    lgl_lgtc,
):
    cho = jnp.zeros((3, 3)).astype("f4")
    cho = jops.index_update(cho, jops.index[0, 0], 10 ** log10_lge_lge)
    cho = jops.index_update(cho, jops.index[1, 1], 10 ** log10_lgl_lgl)
    cho = jops.index_update(cho, jops.index[2, 2], 10 ** log10_lgtc_lgtc)
    cho = jops.index_update(cho, jops.index[1, 0], lge_lgl)
    cho = jops.index_update(cho, jops.index[2, 0], lge_lgtc)
    cho = jops.index_update(cho, jops.index[2, 1], lgl_lgtc)
    cov = jnp.dot(cho, cho.T)
    return cov


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(0, 0, 0, 0, 0, 0)))


@jjit
def frac_late_forming(
    lgm0,
    frac_late_ylo=DEFAULT_MAH_PDF_PARAMS["frac_late_ylo"],
    frac_late_yhi=DEFAULT_MAH_PDF_PARAMS["frac_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, frac_late_ylo, frac_late_yhi)


@jjit
def mean_ue_early_vs_lgm0(
    lgm0,
    mean_ue_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ue_early_ylo"],
    mean_ue_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ue_early_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, mean_ue_early_ylo, mean_ue_early_yhi)


@jjit
def mean_ul_early_vs_lgm0(
    lgm0,
    mean_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ul_early_ylo"],
    mean_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ul_early_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, mean_ul_early_ylo, mean_ul_early_yhi)


@jjit
def mean_lgtc_early_vs_lgm0(
    lgm0,
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, mean_lgtc_early_ylo, mean_lgtc_early_yhi)


@jjit
def mean_ue_late_vs_lgm0(
    lgm0,
    mean_ue_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ue_late_ylo"],
    mean_ue_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ue_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, mean_ue_late_ylo, mean_ue_late_yhi)


@jjit
def mean_ul_late_vs_lgm0(
    lgm0,
    mean_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ul_late_ylo"],
    mean_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ul_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, mean_ul_late_ylo, mean_ul_late_yhi)


@jjit
def mean_lgtc_late_vs_lgm0(
    lgm0,
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, mean_lgtc_late_ylo, mean_lgtc_late_yhi)


@jjit
def chol_ue_ue_early_vs_lgm0(
    lgm0,
    chol_ue_ue_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_early_ylo"],
    chol_ue_ue_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_early_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ue_ue_early_ylo, chol_ue_ue_early_yhi)


@jjit
def chol_ul_ul_early_vs_lgm0(
    lgm0,
    chol_ul_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_early_ylo"],
    chol_ul_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_early_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ul_ul_early_ylo, chol_ul_ul_early_yhi)


@jjit
def chol_lgtc_lgtc_early_vs_lgm0(
    lgm0,
    chol_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_early_ylo"],
    chol_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_early_yhi"],
):
    return _sigmoid(
        lgm0, _LGM_X0, LGM_K, chol_lgtc_lgtc_early_ylo, chol_lgtc_lgtc_early_yhi
    )


@jjit
def chol_ue_ue_late_vs_lgm0(
    lgm0,
    chol_ue_ue_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_late_ylo"],
    chol_ue_ue_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ue_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ue_ue_late_ylo, chol_ue_ue_late_yhi)


@jjit
def chol_ul_ul_late_vs_lgm0(
    lgm0,
    chol_ul_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_late_ylo"],
    chol_ul_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_ul_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ul_ul_late_ylo, chol_ul_ul_late_yhi)


@jjit
def chol_lgtc_lgtc_late_vs_lgm0(
    lgm0,
    chol_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_late_ylo"],
    chol_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_lgtc_lgtc_late_yhi"],
):
    return _sigmoid(
        lgm0, _LGM_X0, LGM_K, chol_lgtc_lgtc_late_ylo, chol_lgtc_lgtc_late_yhi
    )


@jjit
def chol_ue_ul_early_vs_lgm0(
    lgm0,
    chol_ue_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_early_ylo"],
    chol_ue_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_early_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ue_ul_early_ylo, chol_ue_ul_early_yhi)


@jjit
def chol_ue_lgtc_early_vs_lgm0(
    lgm0,
    chol_ue_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_early_ylo"],
    chol_ue_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_early_yhi"],
):
    return _sigmoid(
        lgm0, _LGM_X0, LGM_K, chol_ue_lgtc_early_ylo, chol_ue_lgtc_early_yhi
    )


@jjit
def chol_ul_lgtc_early_vs_lgm0(
    lgm0,
    chol_ul_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_early_ylo"],
    chol_ul_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_early_yhi"],
):
    return _sigmoid(
        lgm0, _LGM_X0, LGM_K, chol_ul_lgtc_early_ylo, chol_ul_lgtc_early_yhi
    )


@jjit
def chol_ue_ul_late_vs_lgm0(
    lgm0,
    chol_ue_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_late_ylo"],
    chol_ue_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_ul_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ue_ul_late_ylo, chol_ue_ul_late_yhi)


@jjit
def chol_ue_lgtc_late_vs_lgm0(
    lgm0,
    chol_ue_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_late_ylo"],
    chol_ue_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ue_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ue_lgtc_late_ylo, chol_ue_lgtc_late_yhi)


@jjit
def chol_ul_lgtc_late_vs_lgm0(
    lgm0,
    chol_ul_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_late_ylo"],
    chol_ul_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["chol_ul_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, chol_ul_lgtc_late_ylo, chol_ul_lgtc_late_yhi)


def get_default_params(lgm):
    frac_late = frac_late_forming(lgm)

    ue_e = mean_ue_early_vs_lgm0(lgm)
    ul_e = mean_ul_early_vs_lgm0(lgm)
    lgtc_e = mean_lgtc_early_vs_lgm0(lgm)

    ue_l = mean_ue_late_vs_lgm0(lgm)
    ul_l = mean_ul_late_vs_lgm0(lgm)
    lgtc_l = mean_lgtc_late_vs_lgm0(lgm)

    lg_ue_ue_e = chol_ue_ue_early_vs_lgm0(lgm)
    lg_ul_ul_e = chol_ul_ul_early_vs_lgm0(lgm)
    lg_lgtc_lgtc_e = chol_lgtc_lgtc_early_vs_lgm0(lgm)

    lg_ue_ue_l = chol_ue_ue_late_vs_lgm0(lgm)
    lg_ul_ul_l = chol_ul_ul_late_vs_lgm0(lgm)
    lg_lgtc_lgtc_l = chol_lgtc_lgtc_late_vs_lgm0(lgm)

    ue_ul_e = chol_ue_ul_early_vs_lgm0(lgm)
    ue_lgtc_e = chol_ue_lgtc_early_vs_lgm0(lgm)
    ul_lgtc_e = chol_ul_lgtc_early_vs_lgm0(lgm)

    ue_ul_l = chol_ue_ul_late_vs_lgm0(lgm)
    ue_lgtc_l = chol_ue_lgtc_late_vs_lgm0(lgm)
    ul_lgtc_l = chol_ul_lgtc_late_vs_lgm0(lgm)

    all_params = (
        frac_late,
        ue_e,
        ul_e,
        lgtc_e,
        lg_ue_ue_e,
        lg_ul_ul_e,
        lg_lgtc_lgtc_e,
        ue_ul_e,
        ue_lgtc_e,
        ul_lgtc_e,
        ue_l,
        ul_l,
        lgtc_l,
        lg_ue_ue_l,
        lg_ul_ul_l,
        lg_lgtc_lgtc_l,
        ue_ul_l,
        ue_lgtc_l,
        ul_lgtc_l,
    )
    return all_params


@jjit
def _get_mah_means_and_covs(
    logmp_arr,
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
    frac_late = frac_late_forming(logmp_arr, frac_late_ylo, frac_late_yhi)
    ue_early, ul_early, lgtc_early = _get_mean_mah_params_early(
        logmp_arr,
        mean_ue_early_ylo,
        mean_ue_early_yhi,
        mean_ul_early_ylo,
        mean_ul_early_yhi,
        mean_lgtc_early_ylo,
        mean_lgtc_early_yhi,
    )
    means_early = jnp.array((ue_early, ul_early, lgtc_early)).T

    covs_early = _get_covs_early(
        logmp_arr,
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
    )

    ue_late, ul_late, lgtc_late = _get_mean_mah_params_late(
        logmp_arr,
        mean_ue_late_ylo,
        mean_ue_late_yhi,
        mean_ul_late_ylo,
        mean_ul_late_yhi,
        mean_lgtc_late_ylo,
        mean_lgtc_late_yhi,
    )
    means_late = jnp.array((ue_late, ul_late, lgtc_late)).T

    covs_late = _get_covs_late(
        logmp_arr,
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
    )
    return frac_late, means_early, covs_early, means_late, covs_late


@jjit
def _get_mean_mah_params_early(
    lgm,
    mean_ue_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ue_early_ylo"],
    mean_ue_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ue_early_yhi"],
    mean_ul_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ul_early_ylo"],
    mean_ul_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ul_early_yhi"],
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
):
    ue = mean_ue_early_vs_lgm0(lgm, mean_ue_early_ylo, mean_ue_early_yhi)
    ul = mean_ul_early_vs_lgm0(lgm, mean_ul_early_ylo, mean_ul_early_yhi)
    lgtc = mean_lgtc_early_vs_lgm0(lgm, mean_lgtc_early_ylo, mean_lgtc_early_yhi)
    return ue, ul, lgtc


@jjit
def _get_mean_mah_params_late(
    lgm,
    mean_ue_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ue_late_ylo"],
    mean_ue_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ue_late_yhi"],
    mean_ul_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_ul_late_ylo"],
    mean_ul_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_ul_late_yhi"],
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
):
    ue = mean_ue_late_vs_lgm0(lgm, mean_ue_late_ylo, mean_ue_late_yhi)
    ul = mean_ul_late_vs_lgm0(lgm, mean_ul_late_ylo, mean_ul_late_yhi)
    lgtc = mean_lgtc_late_vs_lgm0(lgm, mean_lgtc_late_ylo, mean_lgtc_late_yhi)
    return ue, ul, lgtc


@jjit
def _get_covs_early(
    lgmp_arr,
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
):
    _res = _get_chol_params_early(
        lgmp_arr,
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
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_chol_params_early(
    lgm,
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
):
    ue_ue = chol_ue_ue_early_vs_lgm0(lgm, chol_ue_ue_early_ylo, chol_ue_ue_early_yhi)
    ul_ul = chol_ul_ul_early_vs_lgm0(lgm, chol_ul_ul_early_ylo, chol_ul_ul_early_yhi)
    lgtc_lgtc = chol_lgtc_lgtc_early_vs_lgm0(
        lgm, chol_lgtc_lgtc_early_ylo, chol_lgtc_lgtc_early_yhi
    )
    ue_ul = chol_ue_ul_early_vs_lgm0(lgm, chol_ue_ul_early_ylo, chol_ue_ul_early_yhi)
    ue_lgtc = chol_ue_lgtc_early_vs_lgm0(
        lgm, chol_ue_lgtc_early_ylo, chol_ue_lgtc_early_yhi
    )
    ul_lgtc = chol_ul_lgtc_early_vs_lgm0(
        lgm, chol_ul_lgtc_early_ylo, chol_ul_lgtc_early_yhi
    )

    return ue_ue, ul_ul, lgtc_lgtc, ue_ul, ue_lgtc, ul_lgtc


@jjit
def _get_covs_late(
    lgmp_arr,
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
):
    _res = _get_chol_params_late(
        lgmp_arr,
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
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_chol_params_late(
    lgm,
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
):
    ue_ue = chol_ue_ue_late_vs_lgm0(lgm, chol_ue_ue_late_ylo, chol_ue_ue_late_yhi)
    ul_ul = chol_ul_ul_late_vs_lgm0(lgm, chol_ul_ul_late_ylo, chol_ul_ul_late_yhi)
    lgtc_lgtc = chol_lgtc_lgtc_late_vs_lgm0(
        lgm, chol_lgtc_lgtc_late_ylo, chol_lgtc_lgtc_late_yhi
    )
    ue_ul = chol_ue_ul_late_vs_lgm0(lgm, chol_ue_ul_late_ylo, chol_ue_ul_late_yhi)
    ue_lgtc = chol_ue_lgtc_late_vs_lgm0(
        lgm, chol_ue_lgtc_late_ylo, chol_ue_lgtc_late_yhi
    )
    ul_lgtc = chol_ul_lgtc_late_vs_lgm0(
        lgm, chol_ul_lgtc_late_ylo, chol_ul_lgtc_late_yhi
    )

    return ue_ue, ul_ul, lgtc_lgtc, ue_ul, ue_lgtc, ul_lgtc
