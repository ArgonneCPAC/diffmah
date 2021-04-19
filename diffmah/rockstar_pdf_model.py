"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp


MAH_PDF_PARAMS = OrderedDict(
    frac_late_ylo=0.645,
    frac_late_yhi=0.645,
    mean_lge_early_ylo=0.45,
    mean_lge_early_yhi=0.7,
    mean_lgl_early_ylo=-0.9,
    mean_lgl_early_yhi=0.75,
    mean_lgtc_early_ylo=-0.35,
    mean_lgtc_early_yhi=-0.075,
    cov_lge_lge_early_ylo=-0.65,
    cov_lge_lge_early_yhi=-0.835,
    cov_lgl_lgl_early_ylo=0.0,
    cov_lgl_lgl_early_yhi=-1.35,
    cov_lgtc_lgtc_early_ylo=-0.82,
    cov_lgtc_lgtc_early_yhi=-1.0,
    cov_lge_lgl_early_ylo=-0.075,
    cov_lge_lgl_early_yhi=-0.0,
    cov_lge_lgtc_early_ylo=-0.15,
    cov_lge_lgtc_early_yhi=-0.075,
    cov_lgl_lgtc_early_ylo=-0.2,
    cov_lgl_lgtc_early_yhi=-0.075,
    mean_lge_late_ylo=-0.025,
    mean_lge_late_yhi=0.625,
    mean_lgl_late_ylo=-1.5,
    mean_lgl_late_yhi=0.9,
    mean_lgtc_late_ylo=0.475,
    mean_lgtc_late_yhi=0.65,
    cov_lge_lge_late_ylo=-0.85,
    cov_lge_lge_late_yhi=-1.2,
    cov_lgl_lgl_late_ylo=-0.25,
    cov_lgl_lgl_late_yhi=-0.6,
    cov_lgtc_lgtc_late_ylo=-0.8,
    cov_lgtc_lgtc_late_yhi=-0.55,
    cov_lge_lgl_late_ylo=-0.2,
    cov_lge_lgl_late_yhi=0.05,
    cov_lge_lgtc_late_ylo=-0.18,
    cov_lge_lgtc_late_yhi=-0.13,
    cov_lgl_lgtc_late_ylo=0.075,
    cov_lgl_lgtc_late_yhi=-0.125,
)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + jnp.exp(-k * (x - x0)))


def get_default_params(lgm):
    frac_late = frac_late_vs_lgm0(lgm)

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


def frac_late_vs_lgm0(
    lgm0,
    frac_late_ylo=MAH_PDF_PARAMS["frac_late_ylo"],
    frac_late_yhi=MAH_PDF_PARAMS["frac_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, frac_late_ylo, frac_late_yhi)


def mean_lge_early_vs_lgm0(
    lgm0,
    mean_lge_early_ylo=MAH_PDF_PARAMS["mean_lge_early_ylo"],
    mean_lge_early_yhi=MAH_PDF_PARAMS["mean_lge_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lge_early_ylo, mean_lge_early_yhi)


def mean_lgl_early_vs_lgm0(
    lgm0,
    mean_lgl_early_ylo=MAH_PDF_PARAMS["mean_lgl_early_ylo"],
    mean_lgl_early_yhi=MAH_PDF_PARAMS["mean_lgl_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lgl_early_ylo, mean_lgl_early_yhi)


def mean_lgtc_early_vs_lgm0(
    lgm0,
    mean_lgtc_early_ylo=MAH_PDF_PARAMS["mean_lgtc_early_ylo"],
    mean_lgtc_early_yhi=MAH_PDF_PARAMS["mean_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lgtc_early_ylo, mean_lgtc_early_yhi)


def mean_lge_late_vs_lgm0(
    lgm0,
    mean_lge_late_ylo=MAH_PDF_PARAMS["mean_lge_late_ylo"],
    mean_lge_late_yhi=MAH_PDF_PARAMS["mean_lge_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lge_late_ylo, mean_lge_late_yhi)


def mean_lgl_late_vs_lgm0(
    lgm0,
    mean_lgl_late_ylo=MAH_PDF_PARAMS["mean_lgl_late_ylo"],
    mean_lgl_late_yhi=MAH_PDF_PARAMS["mean_lgl_late_yhi"],
):
    return _sigmoid(lgm0, 14, 0.5, mean_lgl_late_ylo, mean_lgl_late_yhi)


def mean_lgtc_late_vs_lgm0(
    lgm0,
    mean_lgtc_late_ylo=MAH_PDF_PARAMS["mean_lgtc_late_ylo"],
    mean_lgtc_late_yhi=MAH_PDF_PARAMS["mean_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, mean_lgtc_late_ylo, mean_lgtc_late_yhi)


def cov_lge_lge_early_vs_lgm0(
    lgm0,
    cov_lge_lge_early_ylo=MAH_PDF_PARAMS["cov_lge_lge_early_ylo"],
    cov_lge_lge_early_yhi=MAH_PDF_PARAMS["cov_lge_lge_early_yhi"],
):
    return _sigmoid(lgm0, 12, 1, cov_lge_lge_early_ylo, cov_lge_lge_early_yhi)


def cov_lgl_lgl_early_vs_lgm0(
    lgm0,
    cov_lgl_lgl_early_ylo=MAH_PDF_PARAMS["cov_lgl_lgl_early_ylo"],
    cov_lgl_lgl_early_yhi=MAH_PDF_PARAMS["cov_lgl_lgl_early_yhi"],
):
    return _sigmoid(lgm0, 12.75, 0.5, cov_lgl_lgl_early_ylo, cov_lgl_lgl_early_yhi)


def cov_lgtc_lgtc_early_vs_lgm0(
    lgm0,
    cov_lgtc_lgtc_early_ylo=MAH_PDF_PARAMS["cov_lgtc_lgtc_early_ylo"],
    cov_lgtc_lgtc_early_yhi=MAH_PDF_PARAMS["cov_lgtc_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgtc_lgtc_early_ylo, cov_lgtc_lgtc_early_yhi)


def cov_lge_lge_late_vs_lgm0(
    lgm0,
    cov_lge_lge_late_ylo=MAH_PDF_PARAMS["cov_lge_lge_late_ylo"],
    cov_lge_lge_late_yhi=MAH_PDF_PARAMS["cov_lge_lge_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lge_late_ylo, cov_lge_lge_late_yhi)


def cov_lgl_lgl_late_vs_lgm0(
    lgm0,
    cov_lgl_lgl_late_ylo=MAH_PDF_PARAMS["cov_lgl_lgl_late_ylo"],
    cov_lgl_lgl_late_yhi=MAH_PDF_PARAMS["cov_lgl_lgl_late_yhi"],
):
    return _sigmoid(lgm0, 14, 1, cov_lgl_lgl_late_ylo, cov_lgl_lgl_late_yhi)


def cov_lgtc_lgtc_late_vs_lgm0(
    lgm0,
    cov_lgtc_lgtc_late_ylo=MAH_PDF_PARAMS["cov_lgtc_lgtc_late_ylo"],
    cov_lgtc_lgtc_late_yhi=MAH_PDF_PARAMS["cov_lgtc_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgtc_lgtc_late_ylo, cov_lgtc_lgtc_late_yhi)


def cov_lge_lgl_early_vs_lgm0(
    lgm0,
    cov_lge_lgl_early_ylo=MAH_PDF_PARAMS["cov_lge_lgl_early_ylo"],
    cov_lge_lgl_early_yhi=MAH_PDF_PARAMS["cov_lge_lgl_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgl_early_ylo, cov_lge_lgl_early_yhi)


def cov_lge_lgtc_early_vs_lgm0(
    lgm0,
    cov_lge_lgtc_early_ylo=MAH_PDF_PARAMS["cov_lge_lgtc_early_ylo"],
    cov_lge_lgtc_early_yhi=MAH_PDF_PARAMS["cov_lge_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgtc_early_ylo, cov_lge_lgtc_early_yhi)


def cov_lgl_lgtc_early_vs_lgm0(
    lgm0,
    cov_lgl_lgtc_early_ylo=MAH_PDF_PARAMS["cov_lgl_lgtc_early_ylo"],
    cov_lgl_lgtc_early_yhi=MAH_PDF_PARAMS["cov_lgl_lgtc_early_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgl_lgtc_early_ylo, cov_lgl_lgtc_early_yhi)


def cov_lge_lgl_late_vs_lgm0(
    lgm0,
    cov_lge_lgl_late_ylo=MAH_PDF_PARAMS["cov_lge_lgl_late_ylo"],
    cov_lge_lgl_late_yhi=MAH_PDF_PARAMS["cov_lge_lgl_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgl_late_ylo, cov_lge_lgl_late_yhi)


def cov_lge_lgtc_late_vs_lgm0(
    lgm0,
    cov_lge_lgtc_late_ylo=MAH_PDF_PARAMS["cov_lge_lgtc_late_ylo"],
    cov_lge_lgtc_late_yhi=MAH_PDF_PARAMS["cov_lge_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lge_lgtc_late_ylo, cov_lge_lgtc_late_yhi)


def cov_lgl_lgtc_late_vs_lgm0(
    lgm0,
    cov_lgl_lgtc_late_ylo=MAH_PDF_PARAMS["cov_lgl_lgtc_late_ylo"],
    cov_lgl_lgtc_late_yhi=MAH_PDF_PARAMS["cov_lgl_lgtc_late_yhi"],
):
    return _sigmoid(lgm0, 13, 0.5, cov_lgl_lgtc_late_ylo, cov_lgl_lgtc_late_yhi)
