"""
"""
import numpy as np
from ..utils import jax_inverse_sigmoid, jax_sigmoid


def test_inverse_sigmoid_actually_inverts():
    """
    """
    x0, k, ylo, yhi = 0, 5, 1, 0
    xarr = np.linspace(-1, 1, 100)
    yarr = np.array(jax_sigmoid(xarr, x0, k, ylo, yhi))
    xarr2 = np.array(jax_inverse_sigmoid(yarr, x0, k, ylo, yhi))
    assert np.allclose(xarr, xarr2, rtol=1e-3)
