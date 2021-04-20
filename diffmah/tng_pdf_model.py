"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax import vmap
from jax.scipy.stats import multivariate_normal as jnorm
from .individual_halo_assembly import DEFAULT_MAH_PARAMS

TODAY = 13.8
LGT0 = jnp.log10(TODAY)


DEFAULT_MAH_PDF_PARAMS = OrderedDict(
    frac_late_ylo=0.55,
    frac_late_yhi=0.75,
    mean_lge_early_ylo=0.30,
    mean_lge_early_yhi=0.74,
    mean_lgl_early_ylo=-1.25,
    mean_lgl_early_yhi=0.90,
    mean_lgtc_early_ylo=-0.47,
    mean_lgtc_early_yhi=-0.03,
    cov_lge_lge_early_ylo=-0.50,
    cov_lge_lge_early_yhi=-0.40,
    cov_lgl_lgl_early_ylo=0.14,
    cov_lgl_lgl_early_yhi=-1.33,
    cov_lgtc_lgtc_early_ylo=-1.00,
    cov_lgtc_lgtc_early_yhi=-1.20,
    cov_lge_lgl_early_ylo=-0.33,
    cov_lge_lgl_early_yhi=0.05,
    cov_lge_lgtc_early_ylo=-0.13,
    cov_lge_lgtc_early_yhi=-0.11,
    cov_lgl_lgtc_early_ylo=0.05,
    cov_lgl_lgtc_early_yhi=-0.35,
    mean_lge_late_ylo=0.05,
    mean_lge_late_yhi=0.64,
    mean_lgl_late_ylo=-1.26,
    mean_lgl_late_yhi=1.30,
    mean_lgtc_late_ylo=0.35,
    mean_lgtc_late_yhi=0.90,
    cov_lge_lge_late_ylo=-1.25,
    cov_lge_lge_late_yhi=-1.00,
    cov_lgl_lgl_late_ylo=-0.38,
    cov_lgl_lgl_late_yhi=-0.41,
    cov_lgtc_lgtc_late_ylo=-1.00,
    cov_lgtc_lgtc_late_yhi=-0.80,
    cov_lge_lgl_late_ylo=-0.35,
    cov_lge_lgl_late_yhi=-0.15,
    cov_lge_lgtc_late_ylo=-0.35,
    cov_lge_lgtc_late_yhi=-0.07,
    cov_lgl_lgtc_late_ylo=0.15,
    cov_lgl_lgtc_late_yhi=-0.05,
)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + jnp.exp(-k * (x - x0)))


def frac_late_forming(
    lgm0,
    frac_late_ylo=DEFAULT_MAH_PDF_PARAMS["frac_late_ylo"],
    frac_late_yhi=DEFAULT_MAH_PDF_PARAMS["frac_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, frac_late_ylo, frac_late_yhi)


def get_default_params(lgm):
    frac_late = frac_late_forming(lgm)

    lge_e = mean_lge_early_vs_lgm0(lgm)
    lgl_e = mean_lgl_early_vs_lgm0(lgm)
    lgtc_e = mean_lgtc_early_vs_lgm0(lgm)

    lge_l = mean_lge_late_vs_lgm0(lgm)
    lgl_l = mean_lgl_late_vs_lgm0(lgm)
    lgtc_l = mean_lgtc_late_vs_lgm0(lgm)

    lg_lge_lge_e = cov_lge_lge_early_vs_lgm0(lgm)
    lg_lgl_lgl_e = cov_lgl_lgl_early_vs_lgm0(lgm)
    lg_lgtc_lgtc_e = cov_lgtc_lgtc_early_vs_lgm0(lgm)

    lg_lge_lge_l = cov_lge_lge_late_vs_lgm0(lgm)
    lg_lgl_lgl_l = cov_lgl_lgl_late_vs_lgm0(lgm)
    lg_lgtc_lgtc_l = cov_lgtc_lgtc_late_vs_lgm0(lgm)

    lge_lgl_e = cov_lge_lgl_early_vs_lgm0(lgm)
    lge_lgtc_e = cov_lge_lgtc_early_vs_lgm0(lgm)
    lgl_lgtc_e = cov_lgl_lgtc_early_vs_lgm0(lgm)

    lge_lgl_l = cov_lge_lgl_late_vs_lgm0(lgm)
    lge_lgtc_l = cov_lge_lgtc_late_vs_lgm0(lgm)
    lgl_lgtc_l = cov_lgl_lgtc_late_vs_lgm0(lgm)

    all_params = (
        frac_late,
        lge_e,
        lgl_e,
        lgtc_e,
        lg_lge_lge_e,
        lg_lgl_lgl_e,
        lg_lgtc_lgtc_e,
        lge_lgl_e,
        lge_lgtc_e,
        lgl_lgtc_e,
        lge_l,
        lgl_l,
        lgtc_l,
        lg_lge_lge_l,
        lg_lgl_lgl_l,
        lg_lgtc_lgtc_l,
        lge_lgl_l,
        lge_lgtc_l,
        lgl_lgtc_l,
    )
    return all_params


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
def _get_cov_early(
    lgmp_arr,
    cov_lge_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_ylo"],
    cov_lge_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_yhi"],
    cov_lgl_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_ylo"],
    cov_lgl_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_yhi"],
    cov_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_ylo"],
    cov_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_yhi"],
    cov_lge_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_ylo"],
    cov_lge_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_yhi"],
    cov_lge_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_ylo"],
    cov_lge_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_yhi"],
    cov_lgl_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_ylo"],
    cov_lgl_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_yhi"],
):
    _res = _get_cov_params_early(
        lgmp_arr,
        cov_lge_lge_early_ylo,
        cov_lge_lge_early_yhi,
        cov_lgl_lgl_early_ylo,
        cov_lgl_lgl_early_yhi,
        cov_lgtc_lgtc_early_ylo,
        cov_lgtc_lgtc_early_yhi,
        cov_lge_lgl_early_ylo,
        cov_lge_lgl_early_yhi,
        cov_lge_lgtc_early_ylo,
        cov_lge_lgtc_early_yhi,
        cov_lgl_lgtc_early_ylo,
        cov_lgl_lgtc_early_yhi,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_cov_params_early(
    lgm,
    cov_lge_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_ylo"],
    cov_lge_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_yhi"],
    cov_lgl_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_ylo"],
    cov_lgl_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_yhi"],
    cov_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_ylo"],
    cov_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_yhi"],
    cov_lge_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_ylo"],
    cov_lge_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_yhi"],
    cov_lge_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_ylo"],
    cov_lge_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_yhi"],
    cov_lgl_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_ylo"],
    cov_lgl_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_yhi"],
):
    lge_lge = cov_lge_lge_early_vs_lgm0(
        lgm, cov_lge_lge_early_ylo, cov_lge_lge_early_yhi
    )
    lgl_lgl = cov_lgl_lgl_early_vs_lgm0(
        lgm, cov_lgl_lgl_early_ylo, cov_lgl_lgl_early_yhi
    )
    lgtc_lgtc = cov_lgtc_lgtc_early_vs_lgm0(
        lgm, cov_lgtc_lgtc_early_ylo, cov_lgtc_lgtc_early_yhi
    )
    lge_lgl = cov_lge_lgl_early_vs_lgm0(
        lgm, cov_lge_lgl_early_ylo, cov_lge_lgl_early_yhi
    )
    lge_lgtc = cov_lge_lgtc_early_vs_lgm0(
        lgm, cov_lge_lgtc_early_ylo, cov_lge_lgtc_early_yhi
    )
    lgl_lgtc = cov_lgl_lgtc_early_vs_lgm0(
        lgm, cov_lgl_lgtc_early_ylo, cov_lgl_lgtc_early_yhi
    )

    return lge_lge, lgl_lgl, lgtc_lgtc, lge_lgl, lge_lgtc, lgl_lgtc


@jjit
def _get_cov_late(
    lgmp_arr,
    cov_lge_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_ylo"],
    cov_lge_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_yhi"],
    cov_lgl_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_ylo"],
    cov_lgl_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_yhi"],
    cov_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_ylo"],
    cov_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_yhi"],
    cov_lge_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_ylo"],
    cov_lge_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_yhi"],
    cov_lge_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_ylo"],
    cov_lge_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_yhi"],
    cov_lgl_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_ylo"],
    cov_lgl_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_yhi"],
):
    _res = _get_cov_params_late(
        lgmp_arr,
        cov_lge_lge_late_ylo,
        cov_lge_lge_late_yhi,
        cov_lgl_lgl_late_ylo,
        cov_lgl_lgl_late_yhi,
        cov_lgtc_lgtc_late_ylo,
        cov_lgtc_lgtc_late_yhi,
        cov_lge_lgl_late_ylo,
        cov_lge_lgl_late_yhi,
        cov_lge_lgtc_late_ylo,
        cov_lge_lgtc_late_yhi,
        cov_lgl_lgtc_late_ylo,
        cov_lgl_lgtc_late_yhi,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_cov_params_late(
    lgm,
    cov_lge_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_ylo"],
    cov_lge_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_yhi"],
    cov_lgl_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_ylo"],
    cov_lgl_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_yhi"],
    cov_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_ylo"],
    cov_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_yhi"],
    cov_lge_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_ylo"],
    cov_lge_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_yhi"],
    cov_lge_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_ylo"],
    cov_lge_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_yhi"],
    cov_lgl_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_ylo"],
    cov_lgl_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_yhi"],
):
    lge_lge = cov_lge_lge_late_vs_lgm0(lgm, cov_lge_lge_late_ylo, cov_lge_lge_late_yhi)
    lgl_lgl = cov_lgl_lgl_late_vs_lgm0(lgm, cov_lgl_lgl_late_ylo, cov_lgl_lgl_late_yhi)
    lgtc_lgtc = cov_lgtc_lgtc_late_vs_lgm0(
        lgm, cov_lgtc_lgtc_late_ylo, cov_lgtc_lgtc_late_yhi
    )
    lge_lgl = cov_lge_lgl_late_vs_lgm0(lgm, cov_lge_lgl_late_ylo, cov_lge_lgl_late_yhi)
    lge_lgtc = cov_lge_lgtc_late_vs_lgm0(
        lgm, cov_lge_lgtc_late_ylo, cov_lge_lgtc_late_yhi
    )
    lgl_lgtc = cov_lgl_lgtc_late_vs_lgm0(
        lgm, cov_lgl_lgtc_late_ylo, cov_lgl_lgtc_late_yhi
    )

    return lge_lge, lgl_lgl, lgtc_lgtc, lge_lgl, lge_lgtc, lgl_lgtc


@jjit
def _get_mean_mah_params_early(
    lgm,
    mean_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_ylo"],
    mean_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_yhi"],
    mean_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_ylo"],
    mean_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_yhi"],
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
):
    lge = mean_lge_early_vs_lgm0(lgm, mean_lge_early_ylo, mean_lge_early_yhi)
    lgl = mean_lgl_early_vs_lgm0(lgm, mean_lgl_early_ylo, mean_lgl_early_yhi)
    lgtc = mean_lgtc_early_vs_lgm0(lgm, mean_lgtc_early_ylo, mean_lgtc_early_yhi)
    return lge, lgl, lgtc


@jjit
def _get_mean_mah_params_late(
    lgm,
    mean_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_ylo"],
    mean_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_yhi"],
    mean_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_ylo"],
    mean_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_yhi"],
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
):
    lge = mean_lge_late_vs_lgm0(lgm, mean_lge_late_ylo, mean_lge_late_yhi)
    lgl = mean_lgl_late_vs_lgm0(lgm, mean_lgl_late_ylo, mean_lgl_late_yhi)
    lgtc = mean_lgtc_late_vs_lgm0(lgm, mean_lgtc_late_ylo, mean_lgtc_late_yhi)
    return lge, lgl, lgtc


def mean_lge_early_vs_lgm0(
    lgm0,
    mean_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_ylo"],
    mean_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lge_early_ylo, mean_lge_early_yhi)


def mean_lgl_early_vs_lgm0(
    lgm0,
    mean_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_ylo"],
    mean_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lgl_early_ylo, mean_lgl_early_yhi)


def mean_lgtc_early_vs_lgm0(
    lgm0,
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lgtc_early_ylo, mean_lgtc_early_yhi)


def mean_lge_late_vs_lgm0(
    lgm0,
    mean_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_ylo"],
    mean_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lge_late_ylo, mean_lge_late_yhi)


def mean_lgl_late_vs_lgm0(
    lgm0,
    mean_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_ylo"],
    mean_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_yhi"],
):
    return _sigmoid(lgm0, 14, 0.5, mean_lgl_late_ylo, mean_lgl_late_yhi)


def mean_lgtc_late_vs_lgm0(
    lgm0,
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lgtc_late_ylo, mean_lgtc_late_yhi)


def cov_lge_lge_early_vs_lgm0(
    lgm0,
    cov_lge_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_ylo"],
    cov_lge_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_yhi"],
):
    return _sigmoid(lgm0, 12, 1, cov_lge_lge_early_ylo, cov_lge_lge_early_yhi)


def cov_lgl_lgl_early_vs_lgm0(
    lgm0,
    cov_lgl_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_ylo"],
    cov_lgl_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_yhi"],
):
    return _sigmoid(lgm0, 12.75, 0.5, cov_lgl_lgl_early_ylo, cov_lgl_lgl_early_yhi)


def cov_lgtc_lgtc_early_vs_lgm0(
    lgm0,
    cov_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_ylo"],
    cov_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgtc_lgtc_early_ylo, cov_lgtc_lgtc_early_yhi)


def cov_lge_lge_late_vs_lgm0(
    lgm0,
    cov_lge_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_ylo"],
    cov_lge_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lge_late_ylo, cov_lge_lge_late_yhi)


def cov_lgl_lgl_late_vs_lgm0(
    lgm0,
    cov_lgl_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_ylo"],
    cov_lgl_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_yhi"],
):
    return _sigmoid(lgm0, 14, 1, cov_lgl_lgl_late_ylo, cov_lgl_lgl_late_yhi)


def cov_lgtc_lgtc_late_vs_lgm0(
    lgm0,
    cov_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_ylo"],
    cov_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgtc_lgtc_late_ylo, cov_lgtc_lgtc_late_yhi)


def cov_lge_lgl_early_vs_lgm0(
    lgm0,
    cov_lge_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_ylo"],
    cov_lge_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgl_early_ylo, cov_lge_lgl_early_yhi)


def cov_lge_lgtc_early_vs_lgm0(
    lgm0,
    cov_lge_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_ylo"],
    cov_lge_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgtc_early_ylo, cov_lge_lgtc_early_yhi)


def cov_lgl_lgtc_early_vs_lgm0(
    lgm0,
    cov_lgl_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_ylo"],
    cov_lgl_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgl_lgtc_early_ylo, cov_lgl_lgtc_early_yhi)


def cov_lge_lgl_late_vs_lgm0(
    lgm0,
    cov_lge_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_ylo"],
    cov_lge_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgl_late_ylo, cov_lge_lgl_late_yhi)


def cov_lge_lgtc_late_vs_lgm0(
    lgm0,
    cov_lge_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_ylo"],
    cov_lge_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgtc_late_ylo, cov_lge_lgtc_late_yhi)


def cov_lgl_lgtc_late_vs_lgm0(
    lgm0,
    cov_lgl_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_ylo"],
    cov_lgl_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgl_lgtc_late_ylo, cov_lgl_lgtc_late_yhi)


@jjit
def _mah_pdf_early(
    lgm,
    lge,
    lgl,
    lgtc,
    mean_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_ylo"],
    mean_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_yhi"],
    mean_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_ylo"],
    mean_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_yhi"],
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
    cov_lge_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_ylo"],
    cov_lge_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_yhi"],
    cov_lgl_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_ylo"],
    cov_lgl_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_yhi"],
    cov_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_ylo"],
    cov_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_yhi"],
    cov_lge_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_ylo"],
    cov_lge_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_yhi"],
    cov_lge_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_ylo"],
    cov_lge_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_yhi"],
    cov_lgl_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_ylo"],
    cov_lgl_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_yhi"],
):
    X = jnp.array((lge, lgl, lgtc)).astype("f4").T
    mu = _get_mean_mah_params_early(
        lgm,
        mean_lge_early_ylo,
        mean_lge_early_yhi,
        mean_lgl_early_ylo,
        mean_lgl_early_yhi,
        mean_lgtc_early_ylo,
        mean_lgtc_early_yhi,
    )
    cov = _get_cov_early(
        lgm,
        cov_lge_lge_early_ylo,
        cov_lge_lge_early_yhi,
        cov_lgl_lgl_early_ylo,
        cov_lgl_lgl_early_yhi,
        cov_lgtc_lgtc_early_ylo,
        cov_lgtc_lgtc_early_yhi,
        cov_lge_lgl_early_ylo,
        cov_lge_lgl_early_yhi,
        cov_lge_lgtc_early_ylo,
        cov_lge_lgtc_early_yhi,
        cov_lgl_lgtc_early_ylo,
        cov_lgl_lgtc_early_yhi,
    )
    return jnorm.pdf(X, mu, cov)


@jjit
def _mah_pdf_late(
    lgm,
    lge,
    lgl,
    lgtc,
    mean_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_ylo"],
    mean_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_yhi"],
    mean_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_ylo"],
    mean_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_yhi"],
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
    cov_lge_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_ylo"],
    cov_lge_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_yhi"],
    cov_lgl_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_ylo"],
    cov_lgl_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_yhi"],
    cov_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_ylo"],
    cov_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_yhi"],
    cov_lge_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_ylo"],
    cov_lge_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_yhi"],
    cov_lge_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_ylo"],
    cov_lge_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_yhi"],
    cov_lgl_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_ylo"],
    cov_lgl_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_yhi"],
):
    X = jnp.array((lge, lgl, lgtc)).astype("f4").T
    mu = _get_mean_mah_params_late(
        lgm,
        mean_lge_late_ylo,
        mean_lge_late_yhi,
        mean_lgl_late_ylo,
        mean_lgl_late_yhi,
        mean_lgtc_late_ylo,
        mean_lgtc_late_yhi,
    )
    cov = _get_cov_late(
        lgm,
        cov_lge_lge_late_ylo,
        cov_lge_lge_late_yhi,
        cov_lgl_lgl_late_ylo,
        cov_lgl_lgl_late_yhi,
        cov_lgtc_lgtc_late_ylo,
        cov_lgtc_lgtc_late_yhi,
        cov_lge_lgl_late_ylo,
        cov_lge_lgl_late_yhi,
        cov_lge_lgtc_late_ylo,
        cov_lge_lgtc_late_yhi,
        cov_lgl_lgtc_late_ylo,
        cov_lgl_lgtc_late_yhi,
    )
    return jnorm.pdf(X, mu, cov)


@jjit
def _get_mah_means_and_covs(
    logmp_arr,
    frac_late_ylo=DEFAULT_MAH_PDF_PARAMS["frac_late_ylo"],
    frac_late_yhi=DEFAULT_MAH_PDF_PARAMS["frac_late_yhi"],
    mean_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_ylo"],
    mean_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_early_yhi"],
    mean_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_ylo"],
    mean_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_early_yhi"],
    mean_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
    cov_lge_lge_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_ylo"],
    cov_lge_lge_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_early_yhi"],
    cov_lgl_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_ylo"],
    cov_lgl_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_early_yhi"],
    cov_lgtc_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_ylo"],
    cov_lgtc_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_early_yhi"],
    cov_lge_lgl_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_ylo"],
    cov_lge_lgl_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_early_yhi"],
    cov_lge_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_ylo"],
    cov_lge_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_early_yhi"],
    cov_lgl_lgtc_early_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_ylo"],
    cov_lgl_lgtc_early_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_early_yhi"],
    mean_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_ylo"],
    mean_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lge_late_yhi"],
    mean_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_ylo"],
    mean_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgl_late_yhi"],
    mean_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
    cov_lge_lge_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_ylo"],
    cov_lge_lge_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lge_late_yhi"],
    cov_lgl_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_ylo"],
    cov_lgl_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgl_late_yhi"],
    cov_lgtc_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_ylo"],
    cov_lgtc_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgtc_lgtc_late_yhi"],
    cov_lge_lgl_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_ylo"],
    cov_lge_lgl_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgl_late_yhi"],
    cov_lge_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_ylo"],
    cov_lge_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lge_lgtc_late_yhi"],
    cov_lgl_lgtc_late_ylo=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_ylo"],
    cov_lgl_lgtc_late_yhi=DEFAULT_MAH_PDF_PARAMS["cov_lgl_lgtc_late_yhi"],
    k=DEFAULT_MAH_PARAMS["mah_k"],
    logtmp=LGT0,
):
    frac_late = frac_late_forming(logmp_arr, frac_late_ylo, frac_late_yhi)
    lge_early, lgl_early, lgtc_early = _get_mean_mah_params_early(
        logmp_arr,
        mean_lge_early_ylo,
        mean_lge_early_yhi,
        mean_lgl_early_ylo,
        mean_lgl_early_yhi,
        mean_lgtc_early_ylo,
        mean_lgtc_early_yhi,
    )
    means_early = jnp.array((lge_early, lgl_early, lgtc_early)).T

    covs_early = _get_cov_early(
        logmp_arr,
        cov_lge_lge_early_ylo,
        cov_lge_lge_early_yhi,
        cov_lgl_lgl_early_ylo,
        cov_lgl_lgl_early_yhi,
        cov_lgtc_lgtc_early_ylo,
        cov_lgtc_lgtc_early_yhi,
        cov_lge_lgl_early_ylo,
        cov_lge_lgl_early_yhi,
        cov_lge_lgtc_early_ylo,
        cov_lge_lgtc_early_yhi,
        cov_lgl_lgtc_early_ylo,
        cov_lgl_lgtc_early_yhi,
    )

    lge_late, lgl_late, lgtc_late = _get_mean_mah_params_late(
        logmp_arr,
        mean_lge_late_ylo,
        mean_lge_late_yhi,
        mean_lgl_late_ylo,
        mean_lgl_late_yhi,
        mean_lgtc_late_ylo,
        mean_lgtc_late_yhi,
    )
    means_late = jnp.array((lge_late, lgl_late, lgtc_late)).T

    covs_late = _get_cov_late(
        logmp_arr,
        cov_lge_lge_late_ylo,
        cov_lge_lge_late_yhi,
        cov_lgl_lgl_late_ylo,
        cov_lgl_lgl_late_yhi,
        cov_lgtc_lgtc_late_ylo,
        cov_lgtc_lgtc_late_yhi,
        cov_lge_lgl_late_ylo,
        cov_lge_lgl_late_yhi,
        cov_lge_lgtc_late_ylo,
        cov_lge_lgtc_late_yhi,
        cov_lgl_lgtc_late_ylo,
        cov_lgl_lgtc_late_yhi,
    )
    return frac_late, means_early, covs_early, means_late, covs_late
