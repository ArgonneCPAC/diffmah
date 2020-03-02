"""Testing functions of the sigmoid-based quenching probability."""
import numpy as np
import pytest

from ..quenching_probability import (
    quenching_prob_cens,
    quenching_prob_sats,
    quenching_prob,
)

NPTS = 4
LOGMHALO = np.linspace(9, 16, NPTS)
QPROB_CENS = np.array([0.15229824, 0.23279159, 0.77519469, 0.89630955])
LOGMPEAK = np.linspace(8, 14, NPTS)
LOGMHOST = np.linspace(11, 17, NPTS)
TINF = np.linspace(-1, 10, NPTS)
QPROB_SATS = np.array([0.150473, 0.27329003, 0.53520561, 0.87461947])


def test_sigmoid_quenching_cens_regression():
    qprob_cens = quenching_prob_cens(LOGMHALO)
    assert np.all(qprob_cens >= 0)
    assert np.all(qprob_cens <= 1)

    assert np.any(qprob_cens > 0)
    assert np.any(qprob_cens < 1)

    assert np.allclose(qprob_cens, QPROB_CENS)


def test_sigmoid_quenching_cens_params():
    qprob_cens = quenching_prob_cens(LOGMHALO)
    qprob_cens2 = quenching_prob_cens(LOGMHALO, fq_cens_logm_crit=11.1)
    assert ~np.allclose(qprob_cens, qprob_cens2)


def test_sigmoid_quenching_sats_regression():
    qprob_sats = quenching_prob_sats(LOGMPEAK, LOGMHOST, TINF)
    assert np.all(qprob_sats >= 0)
    assert np.all(qprob_sats <= 1)

    assert np.any(qprob_sats > 0)
    assert np.any(qprob_sats < 1)

    assert np.allclose(qprob_sats, QPROB_SATS)


def test_sigmoid_quenching_sats_params():
    qrob_sats = quenching_prob_sats(LOGMPEAK, LOGMHOST, TINF)
    qrob_sats2 = quenching_prob_sats(LOGMPEAK, LOGMHOST, TINF, fq_sat_delay_time=3)
    assert not np.all(qrob_sats == qrob_sats2)


def test_sigmoid_quenching_qprob_satcen_consistent():
    """Enforce function responds to its parameters."""

    qprob1 = quenching_prob(-1, LOGMPEAK, LOGMHOST, TINF)
    qprob_cens = quenching_prob_cens(LOGMPEAK)
    assert np.allclose(qprob1, qprob_cens)

    qprob2 = quenching_prob(1, LOGMPEAK, LOGMHOST, TINF)
    qrob_sats = quenching_prob_sats(LOGMPEAK, LOGMHOST, TINF)
    assert np.allclose(qprob2, qrob_sats)
