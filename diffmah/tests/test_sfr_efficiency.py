"""
"""
import numpy as np
from ..sfr_efficiency import log_sfr_efficiency_ms_jax
from ..sfr_efficiency import mean_log_sfr_efficiency_ms_jax
from ..sfr_efficiency import DEFAULT_SFR_MS_PARAMS, MEDIAN_SFR_MS_PARAMS


def test_mean_log_sfr_eff():
    logm0 = 12
    logt = np.linspace(-1, 1.141, 200)
    params = np.array(list(MEDIAN_SFR_MS_PARAMS.values())).astype("f4")
    mean_log_sfr_eff = mean_log_sfr_efficiency_ms_jax(logm0, logt, params)
    assert mean_log_sfr_eff.size == logt.size
    assert np.all(np.isfinite(mean_log_sfr_eff))


def test_log_sfr_eff():
    logt = np.linspace(-1, 1.141, 200)
    params = np.array(list(DEFAULT_SFR_MS_PARAMS.values())).astype("f4")
    log_sfr_eff = log_sfr_efficiency_ms_jax(logt, params)
    assert log_sfr_eff.size == logt.size
    assert np.all(np.isfinite(log_sfr_eff))
