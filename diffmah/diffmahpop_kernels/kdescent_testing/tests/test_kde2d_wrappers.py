"""
"""

import numpy as np
import pytest
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ... import diffmahpop_params as dpp
from .. import kde2d_wrappers as k2w

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
        _preds = k2w.mc_diffmah_preds(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, pred_data)
        for _x in _preds:
            assert np.all(np.isfinite(_x))
        dmhdt_tpt0, log_mah_tpt0, dmhdt_tp, log_mah_tp, ftpt0 = _preds
        assert np.all(dmhdt_tpt0 > 0)
        assert np.all(dmhdt_tp >= 0)
        assert np.all(log_mah_tpt0 < 20)
        assert np.all(log_mah_tp < 20)
        assert np.all(ftpt0 <= 1)
        assert np.all(ftpt0 >= 0)
