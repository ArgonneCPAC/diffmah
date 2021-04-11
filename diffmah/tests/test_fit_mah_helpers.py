"""
"""
import numpy as np
from ..fit_mah_helpers import get_target_data


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
    assert np.allclose(10 ** logt_target, t_sim)
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
    assert np.allclose(t_sim[1:], 10 ** logt_target)
    assert np.allclose(log_mah_sim[1:], log_mah_target, atol=0.01)
