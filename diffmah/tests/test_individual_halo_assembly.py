"""
"""
import numpy as np
from ..individual_halo_assembly import _calc_halo_history, DEFAULT_MAH_PARAMS


def test_calc_halo_history_evaluates():
    tarr = np.linspace(0.1, 14, 500)
    logt = np.log10(tarr)
    logtmp = logt[-1]
    logmp = 12.0
    dmhdt, log_mah = _calc_halo_history(
        logt, logtmp, logmp, *DEFAULT_MAH_PARAMS.values()
    )


def test_rolling_index_agrees_with_hard_coded_expectation():
    pass
