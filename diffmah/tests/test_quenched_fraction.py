"""
"""
import numpy as np
from ..quenched_fraction import quenched_fraction_at_zobs


def test1():
    qfrac = quenched_fraction_at_zobs(12, 0)
    assert 0 <= qfrac <= 1
