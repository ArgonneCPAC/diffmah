"""
"""

from jax import jit as jjit

from . import logm0_pop_early, logm0_pop_late


@jjit
def _pred_logm0_kern_early(logm0_params, lgm_obs, t_obs, t_peak):
    return logm0_pop_early._pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)


@jjit
def _pred_logm0_kern_late(logm0_params, lgm_obs, t_obs, t_peak):
    return logm0_pop_late._pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)
