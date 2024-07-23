"""
"""

import numpy as np
import pytest
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ....diffmah_kernels import DEFAULT_MAH_PARAMS
from ... import diffmahpop_params as dpp
from ... import mc_diffmahpop_kernels as mdk
from .. import kde2d_wrappers as k2w

try:
    import kdescent  # noqa

    HAS_KDESCENT = True
except ImportError:
    HAS_KDESCENT = False

T_MIN_FIT = 0.5


@pytest.mark.skip
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
        _preds = k2w.mc_diffmah_preds(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, pred_data)
        for _x in _preds:
            assert np.all(np.isfinite(_x))
        lgsmar_tpt0, log_mah_tpt0, lgsmar_tp, log_mah_tp, ftpt0 = _preds
        assert np.all(lgsmar_tpt0 < 20)
        assert np.all(lgsmar_tp < 20)
        assert np.all(log_mah_tpt0 < 20)
        assert np.all(log_mah_tp < 20)
        assert np.all(ftpt0 <= 1)
        assert np.all(ftpt0 >= 0)


@pytest.mark.skip
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

        tarr = np.linspace(T_MIN_FIT, t_obs, k2w.N_T_PER_BIN)

        _res = k2w.get_single_sample_self_fit_target_data(
            u_p_fid, tarr, lgm_obs, t_obs, ran_key, lgt0
        )
        X_target, weights_target = _res

        # use default params as the initial guess
        u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(
            dpp.DEFAULT_DIFFMAHPOP_U_PARAMS
        )

        loss_data = tarr, lgm_obs, t_obs, ran_key, lgt0, X_target, weights_target
        loss = k2w.single_sample_kde_loss_self_fit(u_p_init, *loss_data)
        assert np.all(np.isfinite(loss)), (lgm_obs, t_obs)
        assert loss > 0

        loss, grads = k2w.single_sample_kde_loss_and_grad_self_fit(u_p_init, *loss_data)
        assert np.all(np.isfinite(grads)), (lgm_obs, t_obs)


@pytest.mark.skip
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
    _res = k2w.get_multisample_self_fit_target_data(
        u_p_fid, tarr_matrix, lgmobsarr, tobsarr, _keys[:num_samples], lgt0
    )
    X_targets, weights_targets = _res
    loss_data = (
        tarr_matrix,
        lgmobsarr,
        tobsarr,
        _keys[num_samples:],
        lgt0,
        X_targets,
        weights_targets,
    )
    # use default params as the initial guess
    u_p_init = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    loss = k2w.multisample_kde_loss_self_fit(u_p_init, *loss_data)
    assert np.all(np.isfinite(loss))
    assert loss > 0

    loss, grads = k2w.multisample_kde_loss_and_grad_self_fit(u_p_init, *loss_data)
    assert np.all(np.isfinite(grads))


@pytest.mark.skip
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
        _res = k2w.get_multisample_self_fit_target_data(
            u_p_fid, tarr_matrix, lgmobsarr, tobsarr, _keys[:num_samples], lgt0
        )
        X_targets, weights_targets = _res
        loss_data = (
            tarr_matrix,
            lgmobsarr,
            tobsarr,
            _keys[num_samples:],
            lgt0,
            X_targets,
            weights_targets,
        )
        return k2w.multisample_kde_loss_self_fit(u_p, *loss_data)

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


def test_get_single_cen_sample_target_data():
    nhalos = 500
    zz = np.zeros(nhalos)
    mah_params = DEFAULT_MAH_PARAMS._make([zz + p for p in DEFAULT_MAH_PARAMS])
    t_obs = 10.0
    lgm_obs = 11.5
    t_0 = 13.0
    lgt0 = np.log10(t_0)
    n_t = 5
    tarr = np.linspace(0.5, t_obs - 0.01, n_t)
    t_peak = np.random.uniform(2, t_0, nhalos)

    _res = k2w.get_single_cen_sample_target_data(
        mah_params, t_peak, tarr, lgm_obs, t_obs, lgt0
    )
    for _x in _res:
        assert np.all(np.isfinite(_x))
    X_target, weights_target, frac_peaked = _res
    assert frac_peaked.shape == (n_t,)
    assert X_target.shape == (nhalos, 2, n_t)
    assert weights_target.shape == (nhalos, n_t)

    assert np.all(frac_peaked >= 0)
    assert np.all(frac_peaked <= 1)


def test_single_sample_kde_loss_kern():
    ran_key = jran.key(0)
    t_obs = 10.0
    lgm_obs = 11.5
    t_0 = 13.0
    lgt0 = np.log10(t_0)

    n_t = 5
    tarr = np.linspace(0.5, t_obs - 0.01, n_t)

    target_key, pred_key = jran.split(ran_key, 2)
    _res = mdk._mc_diffmah_halo_sample(
        dpp.DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, target_key, lgt0
    )
    (
        mah_params_tpt0,
        mah_params_tp,
        t_peak,
        ftpt0,
        mc_tpt0,
        dmhdt_tpt0,
        log_mah_tpt0,
        dmhdt_tp,
        log_mah_tp,
    ) = _res
    mah_params = [
        np.where(mc_tpt0, x, y) for x, y in zip(mah_params_tpt0, mah_params_tp)
    ]
    mah_params_target = DEFAULT_MAH_PARAMS._make(mah_params)
    t_peak_target = np.where(mc_tpt0, t_0, t_peak)

    _res = k2w.get_single_cen_sample_target_data(
        mah_params_target, t_peak_target, tarr, lgm_obs, t_obs, lgt0
    )
    for _x in _res:
        assert np.all(np.isfinite(_x))
    X_target, weights_target, frac_peaked = _res

    args = (
        dpp.DEFAULT_DIFFMAHPOP_U_PARAMS,
        tarr,
        lgm_obs,
        t_obs,
        pred_key,
        lgt0,
        X_target,
        weights_target,
        frac_peaked,
    )
    loss = k2w.single_sample_kde_loss_kern(*args)
    assert np.all(np.isfinite(loss))
    assert loss > 0
    assert loss < 1e8
