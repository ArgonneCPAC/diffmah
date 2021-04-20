"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from .halo_population_assembly import _get_bimodal_halo_history
from .halo_population_assembly import LGE_ARR, LGL_ARR, X0_ARR

BOUNDS = OrderedDict(
    frac_late_ylo=(0.6, 0.7),
    frac_late_yhi=(0.6, 0.7),
    mean_lge_early_ylo=(0.4, 0.5),
    mean_lge_early_yhi=(0.65, 0.75),
    mean_lgl_early_ylo=(-1, -0.8),
    mean_lgl_early_yhi=(0.7, 0.8),
    mean_lgtc_early_ylo=(-0.5, -0.25),
    mean_lgtc_early_yhi=(-0.25, -0.025),
    cov_lge_lge_early_ylo=(-0.75, -0.5),
    cov_lge_lge_early_yhi=(-1.0, -0.6),
    cov_lgl_lgl_early_ylo=(-0.2, 0.2),
    cov_lgl_lgl_early_yhi=(-1.6, -1.0),
    cov_lgtc_lgtc_early_ylo=(-1.0, -0.6),
    cov_lgtc_lgtc_early_yhi=(-1.2, -0.8),
    cov_lge_lgl_early_ylo=(-0.35, 0.05),
    cov_lge_lgl_early_yhi=(-0.35, 0.05),
    cov_lge_lgtc_early_ylo=(-0.35, 0.05),
    cov_lge_lgtc_early_yhi=(-0.35, 0.05),
    cov_lgl_lgtc_early_ylo=(-0.35, 0.05),
    cov_lgl_lgtc_early_yhi=(-0.35, 0.05),
    mean_lge_late_ylo=(-0.1, 0.05),
    mean_lge_late_yhi=(0.55, 0.7),
    mean_lgl_late_ylo=(-1.8, -1.2),
    mean_lgl_late_yhi=(0.7, 1.1),
    mean_lgtc_late_ylo=(0.35, 0.55),
    mean_lgtc_late_yhi=(0.5, 0.75),
    cov_lge_lge_late_ylo=(-1.0, -0.7),
    cov_lge_lge_late_yhi=(-1.5, -1.0),
    cov_lgl_lgl_late_ylo=(-0.4, -0.05),
    cov_lgl_lgl_late_yhi=(-0.7, -0.5),
    cov_lgtc_lgtc_late_ylo=(-1.0, -0.6),
    cov_lgtc_lgtc_late_yhi=(-0.7, -0.4),
    cov_lge_lgl_late_ylo=(-0.35, -0.05),
    cov_lge_lgl_late_yhi=(-0.15, 0.15),
    cov_lge_lgtc_late_ylo=(-0.35, -0.05),
    cov_lge_lgtc_late_yhi=(-0.25, -0.05),
    cov_lgl_lgtc_late_ylo=(-0.15, 0.15),
    cov_lgl_lgtc_late_yhi=(-0.25, -0.05),
)


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def mse_loss(params, data):
    _lgt, _lgmparr = data[0:2]
    mean_log_mah_targets, var_log_mah_targets = data[2:4]
    mean_dmhdt_targets, var_dmhdt_targets = data[4:]

    _res = _get_bimodal_halo_history(_lgt, _lgmparr, LGE_ARR, LGL_ARR, X0_ARR, *params)
    mean_dmhdt_preds, var_dmhdt_preds = _res[0], _res[3]
    mean_log_mah_preds, var_log_mah_preds = _res[2], _res[4]

    loss_log_mah = _mse(mean_log_mah_preds, mean_log_mah_targets)
    loss_var_log_mah = _mse(var_log_mah_preds, var_log_mah_targets)

    loss_dmhdt = _mse(jnp.log10(mean_dmhdt_preds), jnp.log10(mean_dmhdt_targets))
    loss_var_dmhdt = _mse(jnp.log10(var_dmhdt_preds), jnp.log10(var_dmhdt_targets))

    return loss_log_mah + loss_var_log_mah + loss_dmhdt + loss_var_dmhdt
