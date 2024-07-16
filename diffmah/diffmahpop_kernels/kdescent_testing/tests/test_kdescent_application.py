"""
"""

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jran

from ... import diffmahpop_params as dpp
from .. import dmp_wrappers as dmpw

try:
    import kdescent  # noqa

    HAS_KDESCENT = True
except ImportError:
    HAS_KDESCENT = False


@pytest.mark.skipif("not HAS_KDESCENT")
def test_fit_diffmah_to_itself_with_kdescent():
    ran_key = jran.key(0)

    # Use random diffmahpop parameter to generate fiducial data
    u_p_fid_key, ran_key = jran.split(ran_key, 2)
    n_params = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    uran = jran.uniform(u_p_fid_key, minval=-1.0, maxval=1.0, shape=(n_params,))
    _u_p_list = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
    u_p_fid = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(jnp.array(_u_p_list))

    t_0 = 13.8
    lgt0 = np.log10(t_0)
    lgm_obs = 12.0
    t_obs = 10.0
    num_target_redshifts_per_t_obs = 10
    tarr = np.linspace(0.5, t_obs, num_target_redshifts_per_t_obs)
    pred_data_target = tarr, lgm_obs, t_obs, ran_key, lgt0

    _res = dmpw.mc_diffmah_preds(u_p_fid, pred_data_target)
    log_mah_tpt0, log_mah_tp, ftpt0 = _res
    log_mahs_target = jnp.concatenate((log_mah_tpt0, log_mah_tp))
    weights_target = jnp.concatenate((ftpt0, 1 - ftpt0))

    # use default params as the initial guess
    u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)

    loss_data = tarr, lgm_obs, t_obs, ran_key, lgt0, log_mahs_target, weights_target
    loss = dmpw.single_sample_kde_loss_self_fit(u_p_init, loss_data)
    assert np.all(np.isfinite(loss))
    assert loss > 0

    loss_data = tarr, lgm_obs, t_obs, ran_key, lgt0, log_mahs_target, weights_target
    loss, grads = dmpw.single_sample_kde_loss_and_grad_self_fit(u_p_init, loss_data)
    assert np.all(np.isfinite(grads))
