"""
"""
import numpy as np
from ..quenched_fraction import qprob_at_tobs


def test_qprob_at_tobs_returns_sensible_values():

    for logm in np.linspace(10, 15, 6):
        for tobs in np.linspace(0.1, 13.8, 7):
            qfrac = qprob_at_tobs(12, 13.5)
            assert 0 <= qfrac <= 1


def test_qprob_at_tobs_scales_sensibly_with_mass():
    tobs = 13.5
    logmarr = np.linspace(10, 15, 25)
    qfrac = np.array([qprob_at_tobs(logm0, tobs) for logm0 in logmarr])
    dq = np.diff(qfrac)

    assert np.all(qfrac >= 0)
    assert np.any(qfrac > 0)
    assert np.all(qfrac <= 1)
    assert np.any(qfrac < 1)
    assert np.all(dq > 0)


def test_qprob_at_tobs_scales_sensibly_with_time():
    logm0 = 12
    tarr = np.linspace(1, 13.8, 40)
    qfrac = np.array([qprob_at_tobs(logm0, t) for t in tarr])
    dq = np.diff(qfrac)

    assert np.all(qfrac >= 0)
    assert np.any(qfrac > 0)
    assert np.all(qfrac <= 1)
    assert np.any(qfrac < 1)
    assert np.all(dq >= 0)
    assert not np.allclose(dq, 0)
