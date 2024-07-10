"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..diffmah_kernels import (
    DiffmahParams,
    DiffmahUParams,
    get_bounded_mah_params,
    get_unbounded_mah_params,
)
from . import ftpt0_cens
from .covariance_kernels import _get_diffmahpop_cov
from .diffmahpop_params import get_component_model_params
from .early_index_pop import _pred_early_index_kern
from .late_index_pop import _pred_late_index_kern
from .logm0_kernels.logm0_pop import _pred_logm0_kern
from .logtc_pop import _pred_logtc_kern
from .t_peak_kernels.tp_pdf_cens import mc_tpeak_singlecen

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5


@jjit
def mc_mean_diffmah_params(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    t_0 = 10**lgt0
    model_params = get_component_model_params(diffmahpop_params)
    (
        ftpt0_cens_params,
        tp_pdf_cens_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
        cov_params,
    ) = model_params
    ftpt0 = ftpt0_cens._ftpt0_kernel(ftpt0_cens_params, lgm_obs, t_obs)

    tpc_key, ran_key = jran.split(ran_key, 2)

    lgm_obs = lgm_obs
    t_obs = t_obs
    args = tp_pdf_cens_params, tpc_key, lgm_obs, t_obs, t_0
    t_peak = mc_tpeak_singlecen(*args)

    ftpt0_key, ran_key = jran.split(ran_key, 2)
    mc_tpt0 = jran.uniform(ftpt0_key, shape=()) < ftpt0

    logm0_tpt0 = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_0)
    logm0_tp = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)

    logtc_tpt0 = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_0)
    logtc_tp = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak)

    early_index_tpt0 = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_0)
    early_index_tp = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_peak)

    late_index_tpt0 = _pred_late_index_kern(late_index_params, lgm_obs)
    late_index_tp = _pred_late_index_kern(late_index_params, lgm_obs)

    dmah_tpt0 = DiffmahParams(logm0_tpt0, logtc_tpt0, early_index_tpt0, late_index_tpt0)
    dmah_tp = DiffmahParams(logm0_tp, logtc_tp, early_index_tp, late_index_tp)

    return dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0


@jjit
def mc_diffmah_params_singlecen(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0 = mc_mean_diffmah_params(
        diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0
    )
    u_dmah_tpt0 = get_unbounded_mah_params(dmah_tpt0)
    u_dmah_tp = get_unbounded_mah_params(dmah_tp)

    cov = _get_diffmahpop_cov(diffmahpop_params, lgm_obs)

    ran_key, tpt0_key, tp_key = jran.split(ran_key, 3)
    ran_diffmah_u_params_tpt0 = jran.multivariate_normal(
        tpt0_key, jnp.array(u_dmah_tpt0), cov, shape=()
    )
    ran_diffmah_u_params_tp = jran.multivariate_normal(
        tp_key, jnp.array(u_dmah_tp), cov, shape=()
    )
    ran_diffmah_u_params_tpt0 = DiffmahUParams(*ran_diffmah_u_params_tpt0)
    ran_diffmah_u_params_tp = DiffmahUParams(*ran_diffmah_u_params_tp)

    ran_diffmah_params_tpt0 = get_bounded_mah_params(ran_diffmah_u_params_tpt0)
    ran_diffmah_params_tp = get_bounded_mah_params(ran_diffmah_u_params_tp)
    return ran_diffmah_params_tpt0, ran_diffmah_params_tp, t_peak, ftpt0, mc_tpt0
