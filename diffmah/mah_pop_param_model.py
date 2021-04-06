"""Models for mass-dependence of mean and cov of params early, late, x0"""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import ops as jops
from jax import vmap
from jax.scipy.stats import multivariate_normal as jnorm

FRAC_LATE_FORMING_PARAMS = OrderedDict(
    frac_late_forming_lo=0.45, frac_late_forming_hi=0.58
)

LGE_EARLY_PARAMS = OrderedDict(lge_early_lo=0.425, lge_early_hi=0.875)
LGL_EARLY_PARAMS = OrderedDict(lgl_early_lo=-0.6, lgl_early_hi=0.15)
X0_EARLY_PARAMS = OrderedDict(x0_early=-0.26)
MEAN_PARAMS_EARLY = OrderedDict()
MEAN_PARAMS_EARLY.update(LGE_EARLY_PARAMS)
MEAN_PARAMS_EARLY.update(LGL_EARLY_PARAMS)
MEAN_PARAMS_EARLY.update(X0_EARLY_PARAMS)

LGE_LATE_PARAMS = OrderedDict(lge_late_lo=-0.1, lge_late_hi=0.7)
LGL_LATE_PARAMS = OrderedDict(lgl_late_lo=-1.5, lgl_late_hi=0.4)
X0_LATE_PARAMS = OrderedDict(x0_late=0.55)
MEAN_PARAMS_LATE = OrderedDict()
MEAN_PARAMS_LATE.update(LGE_LATE_PARAMS)
MEAN_PARAMS_LATE.update(LGL_LATE_PARAMS)
MEAN_PARAMS_LATE.update(X0_LATE_PARAMS)


LOG_CHO_LGE_LGE_EARLY_PARAMS = OrderedDict(
    log_cho_lge_lge_early_lo=-0.4, log_cho_lge_lge_early_hi=-0.8
)
LOG_CHO_LGL_LGL_EARLY_PARAMS = OrderedDict(
    log_cho_lgl_lgl_early_lo=-0.25, log_cho_lgl_lgl_early_hi=-1.05
)
LOG_CHO_X0_X0_EARLY_PARAMS = OrderedDict(log_cho_x0_x0_early=-0.85)
CHO_LGE_LGL_EARLY_PARAMS = OrderedDict(cho_lge_lgl_early=-0.08)
CHO_LGE_X0_EARLY_PARAMS = OrderedDict(cho_lge_x0_early=-0.1)
CHO_LGL_X0_EARLY_PARAMS = OrderedDict(cho_lgl_x0_early=-0.08)
COV_PARAMS_EARLY = OrderedDict()
COV_PARAMS_EARLY.update(LOG_CHO_LGE_LGE_EARLY_PARAMS)
COV_PARAMS_EARLY.update(LOG_CHO_LGL_LGL_EARLY_PARAMS)
COV_PARAMS_EARLY.update(LOG_CHO_X0_X0_EARLY_PARAMS)
COV_PARAMS_EARLY.update(CHO_LGE_LGL_EARLY_PARAMS)
COV_PARAMS_EARLY.update(CHO_LGE_X0_EARLY_PARAMS)
COV_PARAMS_EARLY.update(CHO_LGL_X0_EARLY_PARAMS)


LOG_CHO_LGE_LGE_LATE_PARAMS = OrderedDict(
    log_cho_lge_lge_late_lo=-0.675, log_cho_lge_lge_late_hi=-1.125
)
LOG_CHO_LGL_LGL_LATE_PARAMS = OrderedDict(
    log_cho_lgl_lgl_late_lo=-0.3, log_cho_lgl_lgl_late_hi=-0.45
)
LOG_CHO_X0_X0_LATE_PARAMS = OrderedDict(log_cho_x0_x0_late=-0.75)
CHO_LGE_LGL_LATE_PARAMS = OrderedDict(cho_lge_lgl_late=-0.1)
CHO_LGE_X0_LATE_PARAMS = OrderedDict(cho_lge_x0_late=-0.08)
CHO_LGL_X0_LATE_PARAMS = OrderedDict(cho_lgl_x0_late=-0.02)
COV_PARAMS_LATE = OrderedDict()
COV_PARAMS_LATE.update(LOG_CHO_LGE_LGE_LATE_PARAMS)
COV_PARAMS_LATE.update(LOG_CHO_LGL_LGL_LATE_PARAMS)
COV_PARAMS_LATE.update(LOG_CHO_X0_X0_LATE_PARAMS)
COV_PARAMS_LATE.update(CHO_LGE_LGL_LATE_PARAMS)
COV_PARAMS_LATE.update(CHO_LGE_X0_LATE_PARAMS)
COV_PARAMS_LATE.update(CHO_LGL_X0_LATE_PARAMS)


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
    log_cho_lge_lge_early_lo=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=LOG_CHO_X0_X0_EARLY_PARAMS["log_cho_x0_x0_early"],
    cho_lge_lgl_early=CHO_LGE_LGL_EARLY_PARAMS["cho_lge_lgl_early"],
    cho_lge_x0_early=CHO_LGE_X0_EARLY_PARAMS["cho_lge_x0_early"],
    cho_lgl_x0_early=CHO_LGL_X0_EARLY_PARAMS["cho_lgl_x0_early"],
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
    log_cho_lge_lge_late_lo=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=LOG_CHO_X0_X0_LATE_PARAMS["log_cho_x0_x0_late"],
    cho_lge_lgl_late=CHO_LGE_LGL_LATE_PARAMS["cho_lge_lgl_late"],
    cho_lge_x0_late=CHO_LGE_X0_LATE_PARAMS["cho_lge_x0_late"],
    cho_lgl_x0_late=CHO_LGL_X0_LATE_PARAMS["cho_lgl_x0_late"],
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
    log_cho_lge_lge_early_lo=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=LOG_CHO_X0_X0_EARLY_PARAMS["log_cho_x0_x0_early"],
    cho_lge_lgl_early=CHO_LGE_LGL_EARLY_PARAMS["cho_lge_lgl_early"],
    cho_lge_x0_early=CHO_LGE_X0_EARLY_PARAMS["cho_lge_x0_early"],
    cho_lgl_x0_early=CHO_LGL_X0_EARLY_PARAMS["cho_lgl_x0_early"],
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
    log_cho_lge_lge_late_lo=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=LOG_CHO_X0_X0_LATE_PARAMS["log_cho_x0_x0_late"],
    cho_lge_lgl_late=CHO_LGE_LGL_LATE_PARAMS["cho_lge_lgl_late"],
    cho_lge_x0_late=CHO_LGE_X0_LATE_PARAMS["cho_lge_x0_late"],
    cho_lgl_x0_late=CHO_LGL_X0_LATE_PARAMS["cho_lgl_x0_late"],
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
    lge_early_lo=LGE_EARLY_PARAMS["lge_early_lo"],
    lge_early_hi=LGE_EARLY_PARAMS["lge_early_hi"],
    lgl_early_lo=LGL_EARLY_PARAMS["lgl_early_lo"],
    lgl_early_hi=LGL_EARLY_PARAMS["lgl_early_hi"],
    x0_early=X0_EARLY_PARAMS["x0_early"],
):
    lge = _lge_vs_lgm_early(lgm, lge_early_lo, lge_early_hi)
    lgl = _lgl_vs_lgm_early(lgm, lgl_early_lo, lgl_early_hi)
    x0 = _x0_vs_lgm_early(lgm, x0_early)
    return lge, lgl, x0


@jjit
def _get_mean_mah_params_late(
    lgm,
    lge_late_lo=LGE_LATE_PARAMS["lge_late_lo"],
    lge_late_hi=LGE_LATE_PARAMS["lge_late_hi"],
    lgl_late_lo=LGL_LATE_PARAMS["lgl_late_lo"],
    lgl_late_hi=LGL_LATE_PARAMS["lgl_late_hi"],
    x0_late=X0_LATE_PARAMS["x0_late"],
):
    lge = _lge_vs_lgm_late(lgm, lge_late_lo, lge_late_hi)
    lgl = _lgl_vs_lgm_late(lgm, lgl_late_lo, lgl_late_hi)
    x0 = _x0_vs_lgm_late(lgm, x0_late)
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
    cho_lgl_x0_late=CHO_LGL_X0_LATE_PARAMS["cho_lgl_x0_late"],
):
    return jnp.zeros_like(lgm) + cho_lgl_x0_late


@jjit
def _cho_lgl_x0_early(
    lgm,
    cho_lgl_x0_early=CHO_LGL_X0_EARLY_PARAMS["cho_lgl_x0_early"],
):
    return jnp.zeros_like(lgm) + cho_lgl_x0_early


@jjit
def _cho_lge_x0_late(
    lgm,
    cho_lge_x0_late=CHO_LGE_X0_LATE_PARAMS["cho_lge_x0_late"],
):
    return jnp.zeros_like(lgm) + cho_lge_x0_late


@jjit
def _cho_lge_x0_early(
    lgm,
    cho_lge_x0_early=CHO_LGE_X0_EARLY_PARAMS["cho_lge_x0_early"],
):
    return jnp.zeros_like(lgm) + cho_lge_x0_early


@jjit
def _cho_lge_lgl_late(
    lgm,
    cho_lge_lgl_late=CHO_LGE_LGL_LATE_PARAMS["cho_lge_lgl_late"],
):
    return jnp.zeros_like(lgm) + cho_lge_lgl_late


@jjit
def _cho_lge_lgl_early(
    lgm,
    cho_lge_lgl_early=CHO_LGE_LGL_EARLY_PARAMS["cho_lge_lgl_early"],
):
    return jnp.zeros_like(lgm) + cho_lge_lgl_early


@jjit
def _log_cho_x0_x0_early(
    lgm,
    log_cho_x0_x0_early=LOG_CHO_X0_X0_EARLY_PARAMS["log_cho_x0_x0_early"],
):
    return jnp.zeros_like(lgm) + log_cho_x0_x0_early


@jjit
def _log_cho_x0_x0_late(
    lgm,
    log_cho_x0_x0_late=LOG_CHO_X0_X0_LATE_PARAMS["log_cho_x0_x0_late"],
):
    return jnp.zeros_like(lgm) + log_cho_x0_x0_late


@jjit
def _log_cho_lgl_lgl_early(
    lgm,
    log_cho_lgl_lgl_early_lo=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, log_cho_lgl_lgl_early_lo, log_cho_lgl_lgl_early_hi)


@jjit
def _log_cho_lgl_lgl_late(
    lgm,
    log_cho_lgl_lgl_late_lo=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_hi"],
):
    return _sigmoid(lgm, 13.5, 2.0, log_cho_lgl_lgl_late_lo, log_cho_lgl_lgl_late_hi)


@jjit
def _log_cho_lge_lge_late(
    lgm,
    log_cho_lge_lge_late_lo=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, log_cho_lge_lge_late_lo, log_cho_lge_lge_late_hi)


@jjit
def _log_cho_lge_lge_early(
    lgm,
    log_cho_lge_lge_early_lo=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, log_cho_lge_lge_early_lo, log_cho_lge_lge_early_hi)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _lge_vs_lgm_late(
    lgm,
    lge_late_lo=LGE_LATE_PARAMS["lge_late_lo"],
    lge_late_hi=LGE_LATE_PARAMS["lge_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, lge_late_lo, lge_late_hi)


@jjit
def _lge_vs_lgm_early(
    lgm,
    lge_early_lo=LGE_EARLY_PARAMS["lge_early_lo"],
    lge_early_hi=LGE_EARLY_PARAMS["lge_early_hi"],
):
    return _sigmoid(lgm, 13, 0.5, lge_early_lo, lge_early_hi)


@jjit
def _lgl_vs_lgm_late(
    lgm,
    lgl_late_lo=LGL_LATE_PARAMS["lgl_late_lo"],
    lgl_late_hi=LGL_LATE_PARAMS["lgl_late_hi"],
):
    return _sigmoid(lgm, 13, 0.5, lgl_late_lo, lgl_late_hi)


@jjit
def _lgl_vs_lgm_early(
    lgm,
    lgl_early_lo=LGL_EARLY_PARAMS["lgl_early_lo"],
    lgl_early_hi=LGL_EARLY_PARAMS["lgl_early_hi"],
):
    return _sigmoid(lgm, 12.5, 1.5, lgl_early_lo, lgl_early_hi)


@jjit
def _x0_vs_lgm_late(
    lgm,
    x0_late=X0_LATE_PARAMS["x0_late"],
):
    return jnp.zeros_like(lgm) + x0_late


@jjit
def _x0_vs_lgm_early(
    lgm,
    x0_early=X0_EARLY_PARAMS["x0_early"],
):
    return jnp.zeros_like(lgm) + x0_early


@jjit
def _mah_pdf_early(
    lgm,
    lge,
    lgl,
    x0,
    lge_early_lo=LGE_EARLY_PARAMS["lge_early_lo"],
    lge_early_hi=LGE_EARLY_PARAMS["lge_early_hi"],
    lgl_early_lo=LGL_EARLY_PARAMS["lgl_early_lo"],
    lgl_early_hi=LGL_EARLY_PARAMS["lgl_early_hi"],
    x0_early=X0_EARLY_PARAMS["x0_early"],
    log_cho_lge_lge_early_lo=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_lo"],
    log_cho_lge_lge_early_hi=LOG_CHO_LGE_LGE_EARLY_PARAMS["log_cho_lge_lge_early_hi"],
    log_cho_lgl_lgl_early_lo=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_lo"],
    log_cho_lgl_lgl_early_hi=LOG_CHO_LGL_LGL_EARLY_PARAMS["log_cho_lgl_lgl_early_hi"],
    log_cho_x0_x0_early=LOG_CHO_X0_X0_EARLY_PARAMS["log_cho_x0_x0_early"],
    cho_lge_lgl_early=CHO_LGE_LGL_EARLY_PARAMS["cho_lge_lgl_early"],
    cho_lge_x0_early=CHO_LGE_X0_EARLY_PARAMS["cho_lge_x0_early"],
    cho_lgl_x0_early=CHO_LGL_X0_EARLY_PARAMS["cho_lgl_x0_early"],
):
    X = jnp.array((lge, lgl, x0)).astype("f4").T
    mu = _get_mean_mah_params_early(
        lgm,
        lge_early_lo,
        lge_early_hi,
        lgl_early_lo,
        lgl_early_hi,
        x0_early,
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
    lge_late_lo=LGE_LATE_PARAMS["lge_late_lo"],
    lge_late_hi=LGE_LATE_PARAMS["lge_late_hi"],
    lgl_late_lo=LGL_LATE_PARAMS["lgl_late_lo"],
    lgl_late_hi=LGL_LATE_PARAMS["lgl_late_hi"],
    x0_late=X0_LATE_PARAMS["x0_late"],
    log_cho_lge_lge_late_lo=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_lo"],
    log_cho_lge_lge_late_hi=LOG_CHO_LGE_LGE_LATE_PARAMS["log_cho_lge_lge_late_hi"],
    log_cho_lgl_lgl_late_lo=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_lo"],
    log_cho_lgl_lgl_late_hi=LOG_CHO_LGL_LGL_LATE_PARAMS["log_cho_lgl_lgl_late_hi"],
    log_cho_x0_x0_late=LOG_CHO_X0_X0_LATE_PARAMS["log_cho_x0_x0_late"],
    cho_lge_lgl_late=CHO_LGE_LGL_LATE_PARAMS["cho_lge_lgl_late"],
    cho_lge_x0_late=CHO_LGE_X0_LATE_PARAMS["cho_lge_x0_late"],
    cho_lgl_x0_late=CHO_LGL_X0_LATE_PARAMS["cho_lgl_x0_late"],
):
    X = jnp.array((lge, lgl, x0)).astype("f4").T
    mu = _get_mean_mah_params_late(
        lgm,
        lge_late_lo,
        lge_late_hi,
        lgl_late_lo,
        lgl_late_hi,
        x0_late,
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
