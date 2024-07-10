"""
"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..diffmah_kernels import DiffmahParams
from . import ftpt0_cens
from .diffmahpop_params import get_component_model_params
from .early_index_pop import _pred_early_index_kern
from .late_index_pop import _pred_late_index_kern
from .logm0_kernels.logm0_pop import _pred_logm0_kern
from .logtc_pop import _pred_logtc_kern
from .t_peak_kernels.tp_pdf_cens import mc_tpeak_cens

N_TP_PER_HALO = 40
T_OBS_FIT_MIN = 0.5


@jjit
def mc_tp_avg_dmah_params_singlecen(diffmahpop_params, lgm_obs, t_obs, ran_key, lgt0):
    t_0 = 10**lgt0
    model_params = get_component_model_params(diffmahpop_params)
    (
        ftpt0_cens_params,
        tp_pdf_cens_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
    ) = model_params
    ftpt0 = ftpt0_cens._ftpt0_kernel(ftpt0_cens_params, lgm_obs, t_obs)

    tpc_key, ran_key = jran.split(ran_key, 2)
    ZZ = jnp.zeros(N_TP_PER_HALO)

    lgm_obs = lgm_obs + ZZ
    t_obs = t_obs + ZZ
    args = tp_pdf_cens_params, tpc_key, lgm_obs, t_obs, t_0
    t_peak = mc_tpeak_cens(*args)

    ftpt0_key, ran_key = jran.split(ran_key, 2)
    mc_tpt0 = jran.uniform(ftpt0_key, shape=(N_TP_PER_HALO,)) < ftpt0

    logm0_tpt0 = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_0 + ZZ)
    logm0_tp = _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)

    logtc_tpt0 = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_0 + ZZ)
    logtc_tp = _pred_logtc_kern(logtc_params, lgm_obs, t_obs, t_peak)

    early_index_tpt0 = _pred_early_index_kern(
        early_index_params, lgm_obs, t_obs, t_0 + ZZ
    )
    early_index_tp = _pred_early_index_kern(early_index_params, lgm_obs, t_obs, t_peak)

    late_index_tpt0 = _pred_late_index_kern(late_index_params, lgm_obs)
    late_index_tp = _pred_late_index_kern(late_index_params, lgm_obs)

    dmah_tpt0 = DiffmahParams(logm0_tpt0, logtc_tpt0, early_index_tpt0, late_index_tpt0)
    dmah_tp = DiffmahParams(logm0_tp, logtc_tp, early_index_tp, late_index_tp)

    return dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0
