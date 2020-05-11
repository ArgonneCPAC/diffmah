"""
"""
import pytest
import numpy as np
from ..mean_sfr_history import get_mean_galaxy_history


@pytest.mark.xfail
def test_get_galaxy_history():
    tarr = np.linspace(0.1, 14, 500)
    for logm0 in range(10, 16):
        log_sfr, log_sm = get_mean_galaxy_history(logm0, tarr)
        assert log_sfr.size == log_sm.size == tarr.size
