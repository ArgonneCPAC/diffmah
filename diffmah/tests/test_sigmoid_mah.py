"""
"""
import numpy as np
from ..sigmoid_mah import median_logmpeak_from_logt, logmpeak_from_logt


def test_median_logmpeak_from_logt_is_monotonic_at_z0():
    logt = np.log10(np.linspace(0.1, 10 ** 1.14, 1000))
    logmah5 = median_logmpeak_from_logt(logt, 5, logt0=1.14)
    logmah10 = median_logmpeak_from_logt(logt, 10, logt0=1.14)
    logmah15 = median_logmpeak_from_logt(logt, 15, logt0=1.14)
    assert logmah5[-1] < logmah10[-1] < logmah15[-1]
    assert np.all(np.diff(logmah5) > 0)
    assert np.all(np.diff(logmah10) > 0)
    assert np.all(np.diff(logmah15) > 0)


def test_median_logmpeak_from_logt_behaves_properly_with_logt0():
    logt = np.log10(np.linspace(0.1, 10 ** 1.14, 1000))
    logmah10 = median_logmpeak_from_logt(logt, 10, 1)
    logmah10b = median_logmpeak_from_logt(logt, 10, 1.25)
    assert logmah10b[-1] < logmah10[-1]


def test_logmpeak_from_logt_is_monotonic():
    logt = np.log10(np.linspace(0.1, 10 ** 1.14, 1000))
    logtc, k, dlogm_height = 0, 3, 5
    logmpeak_at_logt0, logt0 = 15, 1.14
    logmah = logmpeak_from_logt(logt, logtc, k, dlogm_height, logmpeak_at_logt0, logt0)
    assert np.all(np.diff(logmah) > 0)
