"""
"""
import numpy as np
from ..halo_mass_assembly import halo_logmpeak_vs_time
from ..sigmoid_mah import logmpeak_from_logt as logmpeak_from_logt_np


def test_jax_mah_model_agrees_with_numpy_sigmoid_mah():
    logtk, dlogm_height = 3, 5
    logt0 = 1.14
    t0 = 10 ** logt0
    logtc = 0

    for logm0 in np.linspace(5, 20, 25):
        time = np.linspace(0.1, t0, 25)
        logt = np.log10(time)
        logmpeak_np = logmpeak_from_logt_np(
            logt, logtc, logtk, dlogm_height, logm0, logt0
        )
        logmpeak_jax = halo_logmpeak_vs_time(
            time, logm0, t0=t0, logtc=logtc, logtk=logtk, dlogm_height=dlogm_height,
        )
        assert np.allclose(logmpeak_np, logmpeak_jax, rtol=0.001)
