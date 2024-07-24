"""
"""

import os
from glob import glob

import numpy as np
import pytest
from astropy.cosmology import Planck15
from astropy.table import Table
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
EPS = 1e-3

DATA_DRN = "/Users/aphearin/work/DATA/diffmahpop_data"
CEN_TARGET_FNAMES = sorted(glob(os.path.join(DATA_DRN, "*cen_mah*.h5")))
if len(CEN_TARGET_FNAMES) > 0:
    HAS_TARGET_DATA = True
else:
    HAS_TARGET_DATA = False


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
    tarr = np.linspace(0.5, t_obs - 0.001, n_t)
    t_peak = np.random.uniform(2, t_0, nhalos)

    _res = k2w.get_single_cen_sample_target_data(
        mah_params, t_peak, tarr, lgm_obs, lgt0
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
        mah_params_target, t_peak_target, tarr, lgm_obs, lgt0
    )
    for _x in _res:
        assert np.all(np.isfinite(_x))
    X_target, weights_target, frac_peaked = _res

    ntests = 100
    n_pars = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    for __ in range(ntests):
        pred_key, kde_test_key, param_key = jran.split(pred_key, 3)
        uran = jran.uniform(param_key, minval=-10, maxval=10, shape=(n_pars,))
        u_p = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
        u_params = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(u_p)

        args = (
            u_params,
            tarr,
            lgm_obs,
            t_obs,
            kde_test_key,
            lgt0,
            X_target,
            weights_target,
            frac_peaked,
        )

        loss, grads = k2w.single_sample_kde_loss_and_grad_kern(*args)
        assert np.all(np.isfinite(loss))
        assert loss > 0
        for grad in grads:
            assert np.all(np.isfinite(grad))


@pytest.mark.skipif("not HAS_TARGET_DATA")
def test_multi_cen_sample_kde_loss_kern():
    ran_key = jran.key(0)
    t_0 = 13.8
    lgt0 = np.log10(t_0)
    cen_bnames = [os.path.basename(fn) for fn in CEN_TARGET_FNAMES]

    cen_scale_factors = np.array([float(bn.split("_")[-1][:-3]) for bn in cen_bnames])
    cen_redshifts = 1 / cen_scale_factors - 1.0
    cen_t_obs = Planck15.age(cen_redshifts).value

    t_obs_collector = []
    lgm_obs_collector = []
    tarr_collector = []
    X_target_collector = []
    weights_target_collector = []
    frac_peaked_target_collector = []
    for it_obs, t_obs in enumerate(cen_t_obs):
        cens = Table.read(CEN_TARGET_FNAMES[it_obs], path="data")

        lgm_obs_arr = np.sort(np.unique(cens["lgm_obs"]))
        mah_keys = ("logm0", "logtc", "early_index", "late_index")
        mah_params = DEFAULT_MAH_PARAMS._make([cens[key] for key in mah_keys])

        for im_obs, lgm_obs in enumerate(lgm_obs_arr):
            mmsk = cens["lgm_obs"] == lgm_obs
            mah_params_target = DEFAULT_MAH_PARAMS._make([x[mmsk] for x in mah_params])
            t_peak_target = cens["t_peak"][mmsk]

            tarr = np.linspace(0.5, t_obs - EPS, k2w.N_T_PER_BIN)

            _res = k2w.get_single_cen_sample_target_data(
                mah_params_target, t_peak_target, tarr, lgm_obs, lgt0
            )
            X_target, weights_target, frac_peaked_target = _res
            tarr_collector.append(tarr)
            lgm_obs_collector.append(lgm_obs)
            t_obs_collector.append(t_obs)
            X_target_collector.append(X_target)
            weights_target_collector.append(weights_target)
            frac_peaked_target_collector.append(frac_peaked_target)

    n_samples = len(t_obs_collector)

    tarr_collector = jnp.array(tarr_collector)
    lgm_obs_collector = jnp.array(lgm_obs_collector)
    t_obs_collector = jnp.array(t_obs_collector)
    X_target_collector = jnp.array(X_target_collector)
    weights_target_collector = jnp.array(weights_target_collector)
    frac_peaked_target_collector = jnp.array(frac_peaked_target_collector)

    n_pars = len(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    n_tests = 10
    for __ in range(n_tests):
        ran_key, param_key, test_key = jran.split(ran_key, 3)
        uran = jran.uniform(param_key, minval=-10, maxval=10, shape=(n_pars,))
        u_p = [x + u for x, u in zip(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, uran)]
        u_params = dpp.DEFAULT_DIFFMAHPOP_U_PARAMS._make(u_p)

        test_keys = jran.split(test_key, n_samples)
        args = (
            u_params,
            tarr_collector,
            lgm_obs_collector,
            t_obs_collector,
            test_keys,
            lgt0,
            X_target_collector,
            weights_target_collector,
            frac_peaked_target_collector,
        )
        loss, grads = k2w.multi_sample_kde_loss_and_grad_kern(*args)
        assert np.all(np.isfinite(loss))
        assert loss > 0
        assert np.all(np.isfinite(grads))
