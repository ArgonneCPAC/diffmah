"""
"""

import numpy as np
from jax import random as jran

from ...diffmah_kernels import MAH_PBOUNDS
from .. import mc_diffmahpop_kernels as mcdpk
from ..diffmahpop_params import DEFAULT_DIFFMAHPOP_PARAMS


def test_mc_tp_avg_dmah_params_singlecen():
    lgm_obs = 12.0
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0 = mcdpk.mc_tp_avg_dmah_params_singlecen(
        DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
    )
    assert np.all(t_peak > 0)
    assert np.all(t_peak <= t_0)
    assert np.any(t_peak < t_0)

    assert ftpt0.shape == ()
    assert np.all(ftpt0 >= 0)
    assert np.all(ftpt0 <= 1)
    assert np.any(ftpt0 > 0)
    assert np.any(ftpt0 < 1)

    assert np.allclose(mc_tpt0.mean(), ftpt0, atol=0.1)

    for p, bound in zip(dmah_tpt0, MAH_PBOUNDS):
        assert np.all(bound[0] < p)
        assert np.all(p < bound[1])
    for p, bound in zip(dmah_tp, MAH_PBOUNDS):
        assert np.all(bound[0] < p)
        assert np.all(p < bound[1])


def test_something():
    lgm_obs = 12.0
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    tarr = np.linspace(0.1, t_0, 50)
    args = DEFAULT_DIFFMAHPOP_PARAMS, tarr, lgm_obs, t_obs, ran_key, np.log10(t_0)
    avg_log_mah = mcdpk.mc_tp_avg_mah_singlecen(*args)
    assert avg_log_mah.shape == tarr.shape
    assert np.all(np.isfinite(avg_log_mah))
    assert np.all(avg_log_mah < 15)
    assert np.all(avg_log_mah > 0)
