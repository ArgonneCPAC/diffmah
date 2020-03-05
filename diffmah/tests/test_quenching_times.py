"""
"""
import numpy as np
from ..quenching_times import central_quenching_time, satellite_quenching_time
from ..quenching_times import quenching_function


def test_central_qtime_is_monotonic():
    logm0 = np.linspace(10, 15, 1000)
    qtimes0 = central_quenching_time(logm0, 0.5)
    qtimes1 = central_quenching_time(logm0, np.zeros_like(logm0) + 0.5)
    assert np.allclose(qtimes0, qtimes1)
    assert np.all(np.diff(qtimes0) <= 0)


def test_satellite_qtime_is_earlier_than_central_qtime():
    logm0 = np.linspace(10, 15, 1000)
    qtimes0 = central_quenching_time(logm0, 0.5)
    inftime = np.random.uniform(0, 14, 1000)
    qtimes1 = satellite_quenching_time(logm0, np.zeros_like(logm0) + 0.5, inftime)
    assert np.all(qtimes1 <= qtimes0)


def test_quenching_function():
    t = np.linspace(0, 20, 100)
    assert np.all(quenching_function(t, 10) >= 0)
    assert np.all(quenching_function(t, 10) <= 1)
    dqdt = np.diff(quenching_function(t, 10)) / np.diff(t)
    assert np.all(dqdt <= 0)
    assert np.any(dqdt < 0)
    assert quenching_function(10, 10) == 0.5
