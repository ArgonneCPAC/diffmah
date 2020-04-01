"""
"""
from ..quenched_fraction import qprob_at_tobs


def test1():
    qfrac = qprob_at_tobs(12, 13.5)
    assert 0 <= qfrac <= 1
