"""
"""
from jax import random as jran
import numpy as np
from ..fit_mah_helpers import get_target_data, diffmah_fitter, get_loss_data
from ..monte_carlo_halo_population import mc_halo_population


def test_diffmah_fitter_works_on_diffmah_example():
    ran_key = jran.PRNGKey(0)
    n_halos, n_times = 20, 100
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    lgmp = jran.uniform(ran_key, minval=12, maxval=15, shape=(n_halos,))
    mc_halopop = mc_halo_population(tarr, t0, lgmp)

    LGM_MIN = 10.0
    TOL = 1e-3

    for ihalo in range(n_halos):
        p_init, loss_data = get_loss_data(
            tarr,
            mc_halopop.log_mah[ihalo, :],
            LGM_MIN,
        )

        res = diffmah_fitter(p_init, loss_data, tol=TOL)
        p_best, loss_best = res[:2]
        assert loss_best < TOL


def test_get_target_data_no_cuts():
    t_sim = np.arange(14) + 1
    nt = len(t_sim)
    log_mah_sim = np.linspace(9, 15, nt)
    lgm_min = log_mah_sim[0]
    dlogm_cut = float("inf")
    t_fit_min = -float("inf")
    logt_target, log_mah_target = get_target_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    assert np.allclose(10**logt_target, t_sim)
    assert np.allclose(log_mah_sim, log_mah_target, atol=0.01)


def test_get_target_data_lgm_cut():
    t_sim = np.arange(14) + 1
    nt = len(t_sim)
    log_mah_sim = np.linspace(9, 15, nt)
    lgm_min = log_mah_sim[1]
    dlogm_cut = float("inf")
    t_fit_min = -float("inf")
    logt_target, log_mah_target = get_target_data(
        t_sim, log_mah_sim, lgm_min, dlogm_cut, t_fit_min
    )
    assert logt_target.shape == log_mah_target.shape
    assert np.allclose(t_sim[1:], 10**logt_target)
    assert np.allclose(log_mah_sim[1:], log_mah_target, atol=0.01)
