"""
"""

from jax import random as jran
from jax import value_and_grad, vmap

try:
    import kdescent
except ImportError:
    pass
from jax import jit as jjit
from jax import numpy as jnp

from .. import diffmahpop_params as dpp
from .. import mc_diffmahpop_kernels as mdk

N_T_PER_BIN = 5


@jjit
def mc_diffmah_preds(diffmahpop_u_params, pred_data):
    diffmahpop_params = dpp.get_diffmahpop_params_from_u_params(diffmahpop_u_params)
    tarr, lgm_obs, t_obs, ran_key, lgt0 = pred_data
    _res = mdk._mc_diffmah_halo_sample(
        diffmahpop_params, tarr, lgm_obs, t_obs, ran_key, lgt0
    )
    ftpt0 = _res[3]
    dmhdt_tpt0 = _res[5]
    log_mah_tpt0 = _res[6]
    dmhdt_tp = _res[7]
    log_mah_tp = _res[8]
    return dmhdt_tpt0, log_mah_tpt0, dmhdt_tp, log_mah_tp, ftpt0
