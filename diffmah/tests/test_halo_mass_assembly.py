"""
"""
import numpy as np
from ..halo_mass_assembly import logmpeak_from_logt as logmpeak_from_logt_jax
from ..sigmoid_mah import logmpeak_from_logt as logmpeak_from_logt_np


def test_jax_mah_model_agrees_with_numpy_sigmoid_mah():
    logtk, dlogm_height = 3, 5
    logt0 = 1.14
    logtc = 0

    for logm0 in np.linspace(10, 15, 15):
        logt = np.linspace(-0.5, 1.14, 20)
        logmpeak_np = logmpeak_from_logt_np(
            logt, logtc, logtk, dlogm_height, logm0, logt0
        )
        params = logtc, logtk, dlogm_height, logm0, logt0
        logmpeak_jax = logmpeak_from_logt_jax(logt, params)
        assert np.allclose(logmpeak_jax, logmpeak_np)
