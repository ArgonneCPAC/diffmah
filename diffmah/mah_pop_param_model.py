"""Models for mass-dependence of mean and cov of params early, late, x0"""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import ops as jops
from jax import vmap
from jax.scipy.stats import multivariate_normal as jnorm

FRAC_LATE_FORMING_PARAMS = OrderedDict(
    frac_late_forming_lo=0.45, frac_late_forming_hi=0.65
)
MEAN_PARAMS_EARLY = OrderedDict(
    lge_early_lo=0.48,
    lge_early_hi=0.98,
    lgl_early_lo=-0.60,
    lgl_early_hi=0.22,
    x0_early_lo=-0.30,
    x0_early_hi=-0.21,
)

MEAN_PARAMS_LATE = OrderedDict(
    lge_late_lo=-0.15,
    lge_late_hi=0.76,
    lgl_late_lo=-1.23,
    lgl_late_hi=0.44,
    x0_late_lo=0.42,
    x0_late_hi=0.62,
)
COV_PARAMS_EARLY = OrderedDict(
    log_cho_lge_lge_early_lo=-0.50,
    log_cho_lge_lge_early_hi=-0.79,
    log_cho_lgl_lgl_early_lo=-0.16,
    log_cho_lgl_lgl_early_hi=-1.14,
    log_cho_x0_x0_early=-0.97,
    cho_lge_lgl_early=-0.05,
    cho_lge_x0_early=-0.05,
    cho_lgl_x0_early=-0.11,
)
COV_PARAMS_LATE = OrderedDict(
    log_cho_lge_lge_late_lo=-0.51,
    log_cho_lge_lge_late_hi=-1.34,
    log_cho_lgl_lgl_late_lo=-0.35,
    log_cho_lgl_lgl_late_hi=-0.44,
    log_cho_x0_x0_late=-0.80,
    cho_lge_lgl_late=-0.02,
    cho_lge_x0_late=0.00,
    cho_lgl_x0_late=0.02,
)
DEFAULT_MAH_PDF_PARAMS = OrderedDict()
DEFAULT_MAH_PDF_PARAMS.update(FRAC_LATE_FORMING_PARAMS)
DEFAULT_MAH_PDF_PARAMS.update(MEAN_PARAMS_EARLY)
DEFAULT_MAH_PDF_PARAMS.update(COV_PARAMS_EARLY)
DEFAULT_MAH_PDF_PARAMS.update(MEAN_PARAMS_LATE)
DEFAULT_MAH_PDF_PARAMS.update(COV_PARAMS_LATE)


def _get_cov_scalar(
    log10_lge_lge,
    log10_lgl_lgl,
    log10_x0_x0,
    lge_lgl,
    lge_x0,
    lgl_x0,
):
    cho = jnp.zeros((3, 3)).astype("f4")
    cho = jops.index_update(cho, jops.index[0, 0], 10 ** log10_lge_lge)
    cho = jops.index_update(cho, jops.index[1, 1], 10 ** log10_lgl_lgl)
    cho = jops.index_update(cho, jops.index[2, 2], 10 ** log10_x0_x0)
    cho = jops.index_update(cho, jops.index[1, 0], lge_lgl)
    cho = jops.index_update(cho, jops.index[2, 0], lge_x0)
    cho = jops.index_update(cho, jops.index[2, 1], lgl_x0)
    cov = jnp.dot(cho, cho.T)
    return cov


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(0, 0, 0, 0, 0, 0)))


@jjit
def _get_cov_early(
    lgm,
    log_cho_lge_lge_early_lo=COV_PARAMS_EARLY["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=COV_PARAMS_EARLY["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=COV_PARAMS_EARLY["log_cho_x0_x0_early"],
    cho_lge_lgl_early=COV_PARAMS_EARLY["cho_lge_lgl_early"],
    cho_lge_x0_early=COV_PARAMS_EARLY["cho_lge_x0_early"],
    cho_lgl_x0_early=COV_PARAMS_EARLY["cho_lgl_x0_early"],
):
    _res = _get_cov_mah_params_early(
        lgm,
        log_cho_lge_lge_early_lo,
        log_cho_lge_lge_early_hi,
        log_cho_lgl_lgl_early_lo,
        log_cho_lgl_lgl_early_hi,
        log_cho_x0_x0_early,
        cho_lge_lgl_early,
        cho_lge_x0_early,
        cho_lgl_x0_early,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_cov_late(
    lgm,
    log_cho_lge_lge_late_lo=COV_PARAMS_LATE["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=COV_PARAMS_LATE["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=COV_PARAMS_LATE["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=COV_PARAMS_LATE["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=COV_PARAMS_LATE["log_cho_x0_x0_late"],
    cho_lge_lgl_late=COV_PARAMS_LATE["cho_lge_lgl_late"],
    cho_lge_x0_late=COV_PARAMS_LATE["cho_lge_x0_late"],
    cho_lgl_x0_late=COV_PARAMS_LATE["cho_lgl_x0_late"],
):
    _res = _get_cov_mah_params_late(
        lgm,
        log_cho_lge_lge_late_lo,
        log_cho_lge_lge_late_hi,
        log_cho_lgl_lgl_late_lo,
        log_cho_lgl_lgl_late_hi,
        log_cho_x0_x0_late,
        cho_lge_lgl_late,
        cho_lge_x0_late,
        cho_lgl_x0_late,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_cov_mah_params_early(
    lgm,
    log_cho_lge_lge_early_lo=COV_PARAMS_EARLY["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=COV_PARAMS_EARLY["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=COV_PARAMS_EARLY["log_cho_x0_x0_early"],
    cho_lge_lgl_early=COV_PARAMS_EARLY["cho_lge_lgl_early"],
    cho_lge_x0_early=COV_PARAMS_EARLY["cho_lge_x0_early"],
    cho_lgl_x0_early=COV_PARAMS_EARLY["cho_lgl_x0_early"],
):
    log10_lge_lge = _log_cho_lge_lge_early(
        lgm, log_cho_lge_lge_early_lo, log_cho_lge_lge_early_hi
    )
    log10_lgl_lgl = _log_cho_lgl_lgl_early(
        lgm, log_cho_lgl_lgl_early_lo, log_cho_lgl_lgl_early_hi
    )
    log10_x0_x0 = _log_cho_x0_x0_early(lgm, log_cho_x0_x0_early)

    lge_lgl = _cho_lge_lgl_early(lgm, cho_lge_lgl_early)
    lge_x0 = _cho_lge_x0_early(lgm, cho_lge_x0_early)
    lgl_x0 = _cho_lgl_x0_early(lgm, cho_lgl_x0_early)
    return log10_lge_lge, log10_lgl_lgl, log10_x0_x0, lge_lgl, lge_x0, lgl_x0


@jjit
def _get_cov_mah_params_late(
    lgm,
    log_cho_lge_lge_late_lo=COV_PARAMS_LATE["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=COV_PARAMS_LATE["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=COV_PARAMS_LATE["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=COV_PARAMS_LATE["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=COV_PARAMS_LATE["log_cho_x0_x0_late"],
    cho_lge_lgl_late=COV_PARAMS_LATE["cho_lge_lgl_late"],
    cho_lge_x0_late=COV_PARAMS_LATE["cho_lge_x0_late"],
    cho_lgl_x0_late=COV_PARAMS_LATE["cho_lgl_x0_late"],
):
    log10_lge_lge = _log_cho_lge_lge_late(
        lgm, log_cho_lge_lge_late_lo, log_cho_lge_lge_late_hi
    )
    log10_lgl_lgl = _log_cho_lgl_lgl_late(
        lgm, log_cho_lgl_lgl_late_lo, log_cho_lgl_lgl_late_hi
    )
    log10_x0_x0 = _log_cho_x0_x0_late(lgm, log_cho_x0_x0_late)

    lge_lgl = _cho_lge_lgl_late(lgm, cho_lge_lgl_late)
    lge_x0 = _cho_lge_x0_late(lgm, cho_lge_x0_late)
    lgl_x0 = _cho_lgl_x0_late(lgm, cho_lgl_x0_late)
    return log10_lge_lge, log10_lgl_lgl, log10_x0_x0, lge_lgl, lge_x0, lgl_x0


@jjit
def _get_mean_mah_params_early(
    lgm,
    lge_early_lo=MEAN_PARAMS_EARLY["lge_early_lo"],
    lge_early_hi=MEAN_PARAMS_EARLY["lge_early_hi"],
    lgl_early_lo=MEAN_PARAMS_EARLY["lgl_early_lo"],
    lgl_early_hi=MEAN_PARAMS_EARLY["lgl_early_hi"],
    x0_early_lo=MEAN_PARAMS_EARLY["x0_early_lo"],
    x0_early_hi=MEAN_PARAMS_EARLY["x0_early_hi"],
):
    lge = _lge_vs_lgm_early(lgm, lge_early_lo, lge_early_hi)
    lgl = _lgl_vs_lgm_early(lgm, lgl_early_lo, lgl_early_hi)
    x0 = _x0_vs_lgm_early(lgm, x0_early_lo, x0_early_hi)
    return lge, lgl, x0


@jjit
def _get_mean_mah_params_late(
    lgm,
    lge_late_lo=MEAN_PARAMS_LATE["lge_late_lo"],
    lge_late_hi=MEAN_PARAMS_LATE["lge_late_hi"],
    lgl_late_lo=MEAN_PARAMS_LATE["lgl_late_lo"],
    lgl_late_hi=MEAN_PARAMS_LATE["lgl_late_hi"],
    x0_late_lo=MEAN_PARAMS_LATE["x0_late_lo"],
    x0_late_hi=MEAN_PARAMS_LATE["x0_late_hi"],
):
    lge = _lge_vs_lgm_late(lgm, lge_late_lo, lge_late_hi)
    lgl = _lgl_vs_lgm_late(lgm, lgl_late_lo, lgl_late_hi)
    x0 = _x0_vs_lgm_late(lgm, x0_late_lo, x0_late_hi)
    return lge, lgl, x0


@jjit
def frac_late_forming(
    lgm,
    frac_late_forming_lo=FRAC_LATE_FORMING_PARAMS["frac_late_forming_lo"],
    frac_late_forming_hi=FRAC_LATE_FORMING_PARAMS["frac_late_forming_hi"],
):
    """Fraction of halos with mah_x0 > 0.15"""
    return _sigmoid(lgm, 13, 0.5, frac_late_forming_lo, frac_late_forming_hi)


@jjit
def _cho_lgl_x0_late(
    lgm,
    cho_lgl_x0_late=COV_PARAMS_LATE["cho_lgl_x0_late"],
):
    return jnp.zeros_like(lgm) + cho_lgl_x0_late


@jjit
def _cho_lgl_x0_early(
    lgm,
    cho_lgl_x0_early=COV_PARAMS_EARLY["cho_lgl_x0_early"],
):
    return jnp.zeros_like(lgm) + cho_lgl_x0_early


@jjit
def _cho_lge_x0_late(
    lgm,
    cho_lge_x0_late=COV_PARAMS_LATE["cho_lge_x0_late"],
):
    return jnp.zeros_like(lgm) + cho_lge_x0_late


@jjit
def _cho_lge_x0_early(
    lgm,
    cho_lge_x0_early=COV_PARAMS_EARLY["cho_lge_x0_early"],
):
    return jnp.zeros_like(lgm) + cho_lge_x0_early


@jjit
def _cho_lge_lgl_late(
    lgm,
    cho_lge_lgl_late=COV_PARAMS_LATE["cho_lge_lgl_late"],
):
    return jnp.zeros_like(lgm) + cho_lge_lgl_late


@jjit
def _cho_lge_lgl_early(
    lgm,
    cho_lge_lgl_early=COV_PARAMS_EARLY["cho_lge_lgl_early"],
):
    return jnp.zeros_like(lgm) + cho_lge_lgl_early


@jjit
def _log_cho_x0_x0_early(
    lgm,
    log_cho_x0_x0_early=COV_PARAMS_EARLY["log_cho_x0_x0_early"],
):
    return jnp.zeros_like(lgm) + log_cho_x0_x0_early


@jjit
def _log_cho_x0_x0_late(
    lgm,
    log_cho_x0_x0_late=COV_PARAMS_LATE["log_cho_x0_x0_late"],
):
    return jnp.zeros_like(lgm) + log_cho_x0_x0_late


@jjit
def _log_cho_lgl_lgl_early(
    lgm,
    log_cho_lgl_lgl_early_lo=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, log_cho_lgl_lgl_early_lo, log_cho_lgl_lgl_early_hi)


@jjit
def _log_cho_lgl_lgl_late(
    lgm,
    log_cho_lgl_lgl_late_lo=COV_PARAMS_LATE["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=COV_PARAMS_LATE["log_cho_lgl_lgl_late_hi"],
):
    return _sigmoid(lgm, 13.5, 2.0, log_cho_lgl_lgl_late_lo, log_cho_lgl_lgl_late_hi)


@jjit
def _log_cho_lge_lge_late(
    lgm,
    log_cho_lge_lge_late_lo=COV_PARAMS_LATE["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=COV_PARAMS_LATE["log_cho_lge_lge_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, log_cho_lge_lge_late_lo, log_cho_lge_lge_late_hi)


@jjit
def _log_cho_lge_lge_early(
    lgm,
    log_cho_lge_lge_early_lo=COV_PARAMS_EARLY["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=COV_PARAMS_EARLY["log_cho_lge_lge_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, log_cho_lge_lge_early_lo, log_cho_lge_lge_early_hi)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _lge_vs_lgm_late(
    lgm,
    lge_late_lo=MEAN_PARAMS_LATE["lge_late_lo"],
    lge_late_hi=MEAN_PARAMS_LATE["lge_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, lge_late_lo, lge_late_hi)


@jjit
def _lge_vs_lgm_early(
    lgm,
    lge_early_lo=MEAN_PARAMS_EARLY["lge_early_lo"],
    lge_early_hi=MEAN_PARAMS_EARLY["lge_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, lge_early_lo, lge_early_hi)


@jjit
def _lgl_vs_lgm_late(
    lgm,
    lgl_late_lo=MEAN_PARAMS_LATE["lgl_late_lo"],
    lgl_late_hi=MEAN_PARAMS_LATE["lgl_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, lgl_late_lo, lgl_late_hi)


@jjit
def _lgl_vs_lgm_early(
    lgm,
    lgl_early_lo=MEAN_PARAMS_EARLY["lgl_early_lo"],
    lgl_early_hi=MEAN_PARAMS_EARLY["lgl_early_hi"],
):
    return _sigmoid(lgm, 12.5, 1.5, lgl_early_lo, lgl_early_hi)


@jjit
def _x0_vs_lgm_late(
    lgm,
    x0_late_lo=MEAN_PARAMS_LATE["x0_late_lo"],
    x0_late_hi=MEAN_PARAMS_LATE["x0_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, x0_late_lo, x0_late_hi)


@jjit
def _x0_vs_lgm_early(
    lgm,
    x0_early_lo=MEAN_PARAMS_EARLY["x0_early_lo"],
    x0_early_hi=MEAN_PARAMS_EARLY["x0_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, x0_early_lo, x0_early_hi)


@jjit
def _mah_pdf_early(
    lgm,
    lge,
    lgl,
    x0,
    lge_early_lo=MEAN_PARAMS_EARLY["lge_early_lo"],
    lge_early_hi=MEAN_PARAMS_EARLY["lge_early_hi"],
    lgl_early_lo=MEAN_PARAMS_EARLY["lgl_early_lo"],
    lgl_early_hi=MEAN_PARAMS_EARLY["lgl_early_hi"],
    x0_early_lo=MEAN_PARAMS_EARLY["x0_early_lo"],
    x0_early_hi=MEAN_PARAMS_EARLY["x0_early_hi"],
    log_cho_lge_lge_early_lo=COV_PARAMS_EARLY["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=COV_PARAMS_EARLY["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=COV_PARAMS_EARLY["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=COV_PARAMS_EARLY["log_cho_x0_x0_early"],
    cho_lge_lgl_early=COV_PARAMS_EARLY["cho_lge_lgl_early"],
    cho_lge_x0_early=COV_PARAMS_EARLY["cho_lge_x0_early"],
    cho_lgl_x0_early=COV_PARAMS_EARLY["cho_lgl_x0_early"],
):
    X = jnp.array((lge, lgl, x0)).astype("f4").T
    mu = _get_mean_mah_params_early(
        lgm,
        lge_early_lo,
        lge_early_hi,
        lgl_early_lo,
        lgl_early_hi,
        x0_early_lo,
        x0_early_hi,
    )
    cov = _get_cov_early(
        lgm,
        log_cho_lge_lge_early_lo,
        log_cho_lge_lge_early_hi,
        log_cho_lgl_lgl_early_lo,
        log_cho_lgl_lgl_early_hi,
        log_cho_x0_x0_early,
        cho_lge_lgl_early,
        cho_lge_x0_early,
        cho_lgl_x0_early,
    )
    return jnorm.pdf(X, mu, cov)


@jjit
def _mah_pdf_late(
    lgm,
    lge,
    lgl,
    x0,
    lge_late_lo=MEAN_PARAMS_LATE["lge_late_lo"],
    lge_late_hi=MEAN_PARAMS_LATE["lge_late_hi"],
    lgl_late_lo=MEAN_PARAMS_LATE["lgl_late_lo"],
    lgl_late_hi=MEAN_PARAMS_LATE["lgl_late_hi"],
    x0_late_lo=MEAN_PARAMS_LATE["x0_late_lo"],
    x0_late_hi=MEAN_PARAMS_LATE["x0_late_hi"],
    log_cho_lge_lge_late_lo=COV_PARAMS_LATE["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=COV_PARAMS_LATE["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=COV_PARAMS_LATE["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=COV_PARAMS_LATE["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=COV_PARAMS_LATE["log_cho_x0_x0_late"],
    cho_lge_lgl_late=COV_PARAMS_LATE["cho_lge_lgl_late"],
    cho_lge_x0_late=COV_PARAMS_LATE["cho_lge_x0_late"],
    cho_lgl_x0_late=COV_PARAMS_LATE["cho_lgl_x0_late"],
):
    X = jnp.array((lge, lgl, x0)).astype("f4").T
    mu = _get_mean_mah_params_late(
        lgm,
        lge_late_lo,
        lge_late_hi,
        lgl_late_lo,
        lgl_late_hi,
        x0_late_lo,
        x0_late_hi,
    )
    cov = _get_cov_late(
        lgm,
        log_cho_lge_lge_late_lo,
        log_cho_lge_lge_late_hi,
        log_cho_lgl_lgl_late_lo,
        log_cho_lgl_lgl_late_hi,
        log_cho_x0_x0_late,
        cho_lge_lgl_late,
        cho_lge_x0_late,
        cho_lgl_x0_late,
    )
    return jnorm.pdf(X, mu, cov)
