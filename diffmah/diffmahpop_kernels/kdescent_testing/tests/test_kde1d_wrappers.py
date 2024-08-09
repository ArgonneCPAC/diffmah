"""
"""

import numpy as np
import pytest
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ... import diffmahpop_params as dpp
from .. import kde1d_wrappers as k1w

try:
    import kdescent  # noqa

    HAS_KDESCENT = True
except ImportError:
    HAS_KDESCENT = False

T_MIN_FIT = 0.5


def test_mc_diffmah_preds():
    ran_key = jran.key(0)
    t_0 = 13.8
    lgt0 = np.log10(t_0)
    n_times = 5

    n_tests = 20
    for __ in range(n_tests):
        ran_key, m_key, t_key = jran.split(ran_key, 3)
        lgm_obs = jran.uniform(m_key, minval=9, maxval=16, shape=())
        t_obs = jran.uniform(t_key, minval=3, maxval=t_0, shape=())
        tarr = np.linspace(T_MIN_FIT, t_obs, n_times)
        pred_data = tarr, lgm_obs, t_obs, ran_key, lgt0
        _preds = k1w.mc_diffmah_preds(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, pred_data)
        for _x in _preds:
            assert np.all(np.isfinite(_x))
        log_mah_tpt0, log_mah_tp, ftpt0 = _preds
        assert np.all(log_mah_tpt0 < 20)
        assert np.all(log_mah_tp < 20)
        assert np.all(ftpt0 <= 1)
        assert np.all(ftpt0 >= 0)


@pytest.mark.skipif("not HAS_KDESCENT")
def test_single_sample_kde_loss_self_fit():
    """Enforce that single-sample loss has finite grads"""
    ran_key = jran.key(0)

    t_0 = 13.8
    lgt0 = np.log10(t_0)
    DP = 0.5

    n_tests = 100
    for __ in range(n_tests):

        # Use random diffmahpop parameter to generate fiducial data
        u_p_fid_key, ran_key = jran.split(ran_key, 2)
        n_params = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
        uran = jran.uniform(u_p_fid_key, minval=-DP, maxval=DP, shape=(n_params,))
        _u_p_list = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
        u_p_fid = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(jnp.array(_u_p_list))

        ran_key, lgm_key, t_obs_key = jran.split(ran_key, 3)
        lgm_obs = jran.uniform(lgm_key, minval=11, maxval=15, shape=())
        t_obs = jran.uniform(t_obs_key, minval=4, maxval=t_0, shape=())

        tarr = np.linspace(T_MIN_FIT, t_obs, k1w.N_T_PER_BIN)

        _res = k1w.get_single_sample_self_fit_target_data(
            u_p_fid, tarr, lgm_obs, t_obs, ran_key, lgt0
        )
        log_mahs_target, weights_target = _res

        # use default params as the initial guess
        u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(
            dpp.DEFAULT_DIFFMAHPOP_U_PARAMS
        )

        loss_data = tarr, lgm_obs, t_obs, ran_key, lgt0, log_mahs_target, weights_target
        loss = k1w.single_sample_kde_loss_self_fit(u_p_init, *loss_data)
        assert np.all(np.isfinite(loss)), (lgm_obs, t_obs)
        assert loss > 0

        loss, grads = k1w.single_sample_kde_loss_and_grad_self_fit(u_p_init, *loss_data)
        assert np.all(np.isfinite(grads)), (lgm_obs, t_obs)


@pytest.mark.skipif("not HAS_KDESCENT")
def test_multisample_kde_loss_self_fit():
    """Enforce that multi-sample loss has finite grads"""
    ran_key = jran.key(0)

    # Use random diffmahpop parameter to generate fiducial data
    DP = 0.1
    u_p_fid_key, ran_key = jran.split(ran_key, 2)
    n_params = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    uran = jran.uniform(u_p_fid_key, minval=-DP, maxval=DP, shape=(n_params,))
    _u_p_list = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
    u_p_fid = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(jnp.array(_u_p_list))

    lgmobs_key, tobs_key, ran_key = jran.split(ran_key, 3)
    num_samples = 5
    t_0 = 13.8
    lgt0 = np.log10(t_0)
    lgmobsarr = jran.uniform(lgmobs_key, minval=11, maxval=15, shape=(num_samples,))
    tobsarr = jran.uniform(tobs_key, minval=4, maxval=13, shape=(num_samples,))
    num_target_redshifts_per_t_obs = 10

    tarr_matrix = jnp.array(
        [jnp.linspace(T_MIN_FIT, t, num_target_redshifts_per_t_obs) for t in tobsarr]
    )
    _keys = jran.split(ran_key, num_samples * 2)
    _res = k1w.get_multisample_self_fit_target_data(
        u_p_fid, tarr_matrix, lgmobsarr, tobsarr, _keys[:num_samples], lgt0
    )
    log_mahs_targets, weights_targets = _res
    loss_data = (
        tarr_matrix,
        lgmobsarr,
        tobsarr,
        _keys[num_samples:],
        lgt0,
        log_mahs_targets,
        weights_targets,
    )
    # use default params as the initial guess
    u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    loss = k1w.multisample_kde_loss_self_fit(u_p_init, *loss_data)
    assert np.all(np.isfinite(loss))
    assert loss > 0

    loss, grads = k1w.multisample_kde_loss_and_grad_self_fit(u_p_init, *loss_data)
    assert np.all(np.isfinite(grads))


@pytest.mark.skipif("not HAS_KDESCENT")
def test_kdescent_adam_self_fit():
    """Enforce that kdescent.adam terminates without NaNs"""
    ran_key = jran.key(0)

    # Use random diffmahpop parameter to generate fiducial data
    DP = 0.5
    u_p_fid_key, ran_key = jran.split(ran_key, 2)
    n_params = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    uran = jran.uniform(u_p_fid_key, minval=-DP, maxval=DP, shape=(n_params,))
    _u_p_list = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
    u_p_fid = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(jnp.array(_u_p_list))

    lgmobs_key, tobs_key, ran_key = jran.split(ran_key, 3)
    num_samples = 5
    t_0 = 13.8
    lgt0 = np.log10(t_0)
    lgmobsarr = jran.uniform(lgmobs_key, minval=11, maxval=15, shape=(num_samples,))
    tobsarr = jran.uniform(tobs_key, minval=4, maxval=13, shape=(num_samples,))
    num_target_redshifts_per_t_obs = 10

    tarr_matrix = jnp.array(
        [jnp.linspace(T_MIN_FIT, t, num_target_redshifts_per_t_obs) for t in tobsarr]
    )

    @jjit
    def kde_loss(u_p, randkey):
        _keys = jran.split(randkey, num_samples * 2)
        _res = k1w.get_multisample_self_fit_target_data(
            u_p_fid, tarr_matrix, lgmobsarr, tobsarr, _keys[:num_samples], lgt0
        )
        log_mahs_targets, weights_targets = _res
        loss_data = (
            tarr_matrix,
            lgmobsarr,
            tobsarr,
            _keys[num_samples:],
            lgt0,
            log_mahs_targets,
            weights_targets,
        )
        return k1w.multisample_kde_loss_self_fit(u_p, *loss_data)

    # use default params as the initial guess
    u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)

    adam_results = kdescent.adam(
        kde_loss,
        u_p_init,
        nsteps=5,
        learning_rate=0.1,
        randkey=12345,
    )
    u_p_best = adam_results[-1]
    assert np.all(np.isfinite(u_p_best))
