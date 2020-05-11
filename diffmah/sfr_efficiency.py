"""
"""
from collections import OrderedDict
from .utils import jax_sigmoid, _get_param_dict
from jax import jit as jax_jit
from jax import vmap as jax_vmap

MEDIAN_SFR_MS_PARAMS = OrderedDict(
    lge0_lgmc=13.2,
    lge0_at_lgmc=-1.5,
    lge0_early_slope=0.45,
    lge0_late_slope=-1.0,
    k_early_lgmc=11.75,
    k_early_k=3,
    k_early_ylo=5,
    k_early_yhi=9.5,
    lgtc_x0=13.35,
    lgtc_k=0.75,
    lgtc_ylo=1.25,
    lgtc_yhi=-0.65,
    lgec_x0=12.65,
    lgec_k=5,
    lgec_ylo=-0.3,
    lgec_yhi=-0.725,
    k_trans_c0=7,
    a_late_x0=13.5,
    a_late_k=1,
    a_late_ylo=0,
    a_late_yhi=-5,
)

DEFAULT_SFR_MS_PARAMS = OrderedDict(
    lge0=-1.25, k_early=4, lgtc=0.5, lgec=-0.5, k_trans=7, a_late=-3
)


def mean_log_sfr_efficiency_ms_jax(logm0, logt, mean_sfr_eff_params):
    sfr_eff_params = _get_median_growth_params(logm0, *mean_sfr_eff_params)
    return _log_sfr_efficiency_ms_jax(logt, *sfr_eff_params)


def log_sfr_efficiency_ms_jax(logt, sfr_eff_params):
    return _log_sfr_efficiency_ms_jax(logt, *sfr_eff_params)


def _log_sfr_efficiency_ms_jax_kern(logt, lge0, k_early, lgtc, lgec, k_trans, a_late):
    dy = lgec - lge0
    epsilon_early = jax_sigmoid(logt, lgtc, k_early, lge0, lge0 + 2 * dy)
    epsilon_late = a_late * (logt - lgtc) + lgec - 1 / 4
    return jax_sigmoid(logt, lgtc, k_trans, epsilon_early, epsilon_late)


_log_sfr_efficiency_ms_jax = jax_jit(
    jax_vmap(
        _log_sfr_efficiency_ms_jax_kern, in_axes=(0, None, None, None, None, None, None)
    )
)


def _get_median_growth_params(
    logm,
    lge0_lgmc,
    lge0_at_lgmc,
    lge0_early_slope,
    lge0_late_slope,
    k_early_lgmc,
    k_early_k,
    k_early_ylo,
    k_early_yhi,
    lgtc_x0,
    lgtc_k,
    lgtc_ylo,
    lgtc_yhi,
    lgec_x0,
    lgec_k,
    lgec_ylo,
    lgec_yhi,
    k_trans_c0,
    a_late_x0,
    a_late_k,
    a_late_ylo,
    a_late_yhi,
):
    lge0 = _lge0_vs_lgm0_kern(
        logm, lge0_lgmc, lge0_at_lgmc, lge0_early_slope, lge0_late_slope
    )
    k_early = _k_early_vs_lgm0_kern(
        logm, k_early_lgmc, k_early_k, k_early_ylo, k_early_yhi
    )
    lgtc = _lgtc_vs_lgm0_kern(logm, lgtc_x0, lgtc_k, lgtc_ylo, lgtc_yhi)
    lgec = _lgec_vs_lgm0_kern(logm, lgec_x0, lgec_k, lgec_ylo, lgec_yhi)
    k_trans = _k_trans_vs_lgm0_kern(logm, k_trans_c0)
    a_late = _a_late_vs_lgm0_kern(logm, a_late_x0, a_late_k, a_late_ylo, a_late_yhi)
    return lge0, k_early, lgtc, lgec, k_trans, a_late


def _lge0_vs_lgm0_kern(
    logm, lge0_lgmc, lge0_at_lgmc, lge0_early_slope, lge0_late_slope
):
    ylo = lge0_at_lgmc + (logm - lge0_lgmc) * lge0_early_slope
    yhi = lge0_at_lgmc + (logm - lge0_lgmc) * lge0_late_slope
    return jax_sigmoid(logm, lge0_lgmc, 5, ylo, yhi)


def _k_early_vs_lgm0_kern(logm, k_early_lgmc, k_early_k, k_early_ylo, k_early_yhi):
    return jax_sigmoid(logm, k_early_lgmc, k_early_k, k_early_ylo, k_early_yhi)


def _lgtc_vs_lgm0_kern(logm, lgtc_x0, lgtc_k, lgtc_ylo, lgtc_yhi):
    return jax_sigmoid(logm, lgtc_x0, lgtc_k, lgtc_ylo, lgtc_yhi)


def _lgec_vs_lgm0_kern(logm, lgec_x0, lgec_k, lgec_ylo, lgec_yhi):
    return jax_sigmoid(logm, lgec_x0, lgec_k, lgec_ylo, lgec_yhi)


def _k_trans_vs_lgm0_kern(logm, k_trans_c0):
    return logm - logm + k_trans_c0


def _a_late_vs_lgm0_kern(logm, a_late_x0, a_late_k, a_late_ylo, a_late_yhi):
    return jax_sigmoid(logm, a_late_x0, a_late_k, a_late_ylo, a_late_yhi)
