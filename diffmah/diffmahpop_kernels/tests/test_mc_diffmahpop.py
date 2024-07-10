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
    dmah_tpt0, dmah_tp, t_peak, ftpt0, mc_tpt0 = mcdpk.mc_mean_diffmah_params(
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

    for p, bound in zip(dmah_tpt0, MAH_PBOUNDS):
        assert np.all(bound[0] < p)
        assert np.all(p < bound[1])
    for p, bound in zip(dmah_tp, MAH_PBOUNDS):
        assert np.all(bound[0] < p)
        assert np.all(p < bound[1])


def test_mc_diffmah_params_singlecen():
    lgm_obs = 12.0
    t_obs = 10.0
    t_0 = 13.8
    ran_key = jran.key(0)
    args = DEFAULT_DIFFMAHPOP_PARAMS, lgm_obs, t_obs, ran_key, np.log10(t_0)
    _res = mcdpk.mc_mean_diffmah_params(*args)
    ran_diffmah_params_tpt0, ran_diffmah_params_tp, t_peak, ftpt0, mc_tpt0 = _res
    for _x in _res:
        assert np.all(np.isfinite(_x))
