"""
"""
import numpy as np
from ..quenching_history import log_ms_fraction_um_median


def test_log_ms_fraction_um_median1():
    lgtarr = np.linspace(0, 1.15, 500)
    [log_ms_fraction_um_median(lgm, lgtarr) for lgm in np.arange(10, 15, 0.5)]
