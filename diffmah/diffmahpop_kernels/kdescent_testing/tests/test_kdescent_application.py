"""
"""

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jran

from ... import diffmahpop_params as dpp
from ..dmp_wrappers import mc_diffmah_preds

try:
    import kdescent  # noqa

    HAS_KDESCENT = True
except ImportError:
    HAS_KDESCENT = False


@pytest.mark.skipif("not HAS_KDESCENT")
def test_fit_diffmah_to_itself_with_kdescent():
    ran_key = jran.key(0)

    # use default params as the initial guess
    u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)

    # Use randomly diffmahpop parameter to generate fiducial data
    u_p_fid_key, ran_key = jran.split(ran_key, 2)
    n_params = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    uran = jran.uniform(u_p_fid_key, minval=-0.01, maxval=0.01, shape=(n_params,))
    _u_p_list = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
    u_p_fid = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(jnp.array(_u_p_list))

    t_0 = 13.8
    lgt0 = np.log10(t_0)
    lgm_obs = 12.0
    t_obs = 10.0
    num_target_redshifts_per_t_obs = 10
    tarr = np.linspace(0.5, t_obs, num_target_redshifts_per_t_obs)
    pred_data = tarr, lgm_obs, t_obs, ran_key, lgt0

    _res = mc_diffmah_preds(u_p_init, pred_data)
    log_mah_tpt0, log_mah_tp, ftpt0 = _res
    log_mahs_pred = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    weights_pred = jnp.concatenate((ftpt0, 1 - ftpt0))

    _res = mc_diffmah_preds(u_p_fid, pred_data)
    log_mah_tpt0, log_mah_tp, ftpt0 = _res
    log_mahs_target = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    weights_target = jnp.concatenate((ftpt0, 1 - ftpt0))

    kcalc = kdescent.KCalc(log_mahs_target, weights_target)

    model_counts, truth_counts = kcalc.compare_kde_counts(
        ran_key, log_mahs_pred, weights_pred
    )
    diff = (model_counts - truth_counts) ** 2
    loss = jnp.mean(diff)
    assert loss > 0
    assert loss < 1e10
