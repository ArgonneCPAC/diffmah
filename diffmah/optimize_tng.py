"""
"""
from collections import OrderedDict

from jax import jit as jjit
from jax import numpy as jnp

from .halo_population_assembly import (
    LGTC_ARR,
    UE_ARR,
    UL_ARR,
    _get_bimodal_halo_history,
)

BOUNDS = OrderedDict(
    frac_late_ylo=(0.3, 0.6),
    frac_late_yhi=(0.6, 0.8),
    mean_ue_early_ylo=(0.35, 0.85),
    mean_ue_early_yhi=(3.5, 4.25),
    mean_ul_early_ylo=(-0.8, -0.25),
    mean_ul_early_yhi=(-0.8, -0.25),
    mean_lgtc_early_ylo=(-1.0, -0.35),
    mean_lgtc_early_yhi=(0.75, 1.25),
    chol_ue_ue_early_ylo=(0.0, 0.1),
    chol_ue_ue_early_yhi=(-0.25, 0.0),
    chol_ul_ul_early_ylo=(-0.4, -0.1),
    chol_ul_ul_early_yhi=(-0.15, 0.05),
    chol_lgtc_lgtc_early_ylo=(-0.6, -0.3),
    chol_lgtc_lgtc_early_yhi=(-1.5, -1.0),
    chol_ue_ul_early_ylo=(-0.75, -0.5),
    chol_ue_ul_early_yhi=(-0.65, -0.45),
    chol_ue_lgtc_early_ylo=(-0.15, 0.0),
    chol_ue_lgtc_early_yhi=(-0.15, 0.0),
    chol_ul_lgtc_early_ylo=(-0.2, 0.0),
    chol_ul_lgtc_early_yhi=(-0.25, -0.1),
    mean_ue_late_ylo=(0.3, 0.6),
    mean_ue_late_yhi=(2.6, 3.1),
    mean_ul_late_ylo=(-3.05, -2.5),
    mean_ul_late_yhi=(-2.2, -1.4),
    mean_lgtc_late_ylo=(-0.1, 0.25),
    mean_lgtc_late_yhi=(1.6, 2.15),
    chol_ue_ue_late_ylo=(-0.2, 0.0),
    chol_ue_ue_late_yhi=(-0.9, -0.6),
    chol_ul_ul_late_ylo=(-0.1, 0.3),
    chol_ul_ul_late_yhi=(-0.4, -0.1),
    chol_lgtc_lgtc_late_ylo=(-0.5, -0.15),
    chol_lgtc_lgtc_late_yhi=(-1.35, -0.85),
    chol_ue_ul_late_ylo=(-1.4, -1.0),
    chol_ue_ul_late_yhi=(0.25, 0.65),
    chol_ue_lgtc_late_ylo=(-0.2, 0.0),
    chol_ue_lgtc_late_yhi=(-0.15, 0.04),
    chol_ul_lgtc_late_ylo=(0.25, 0.6),
    chol_ul_lgtc_late_yhi=(0.15, 0.4),
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

    _res = _get_bimodal_halo_history(_lgt, _lgmparr, UE_ARR, UL_ARR, LGTC_ARR, *params)
    mean_dmhdt_preds, var_dmhdt_preds = _res[0], _res[3]
    mean_log_mah_preds, var_log_mah_preds = _res[2], _res[4]

    loss_log_mah = _mse(mean_log_mah_preds, mean_log_mah_targets)
    loss_var_log_mah = _mse(var_log_mah_preds, var_log_mah_targets)

    loss_dmhdt = _mse(jnp.log10(mean_dmhdt_preds), jnp.log10(mean_dmhdt_targets))
    loss_var_dmhdt = _mse(jnp.log10(var_dmhdt_preds), jnp.log10(var_dmhdt_targets))

    return loss_log_mah + loss_var_log_mah + loss_dmhdt + loss_var_dmhdt
