"""
"""
import pytest
from ..quenched_fraction import quenched_fraction_at_tobs


@pytest.mark.xfail
def test1():
    qfrac = quenched_fraction_at_tobs(12, 0)
    assert 0 <= qfrac <= 1
