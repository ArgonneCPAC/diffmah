"""
"""
import numpy as np
from scipy.integrate import trapz
from ..halo_mass_assembly import halo_mass_vs_time, halo_dmdt_vs_time
from ..sigmoid_mah import logmpeak_from_logt as logmpeak_from_logt_np


def test_jax_mah_model_agrees_with_numpy_sigmoid_mah():
    """Enforce new implementation agrees with old."""
    logtk, dlogm_height = 3, 5
    logt0 = 1.14
    t0 = 10 ** logt0
    logtc = 0

    for logm0 in np.linspace(5, 20, 25):
        time = np.linspace(0.1, t0, 35)
        logt = np.log10(time)

        logmpeak_np = logmpeak_from_logt_np(
            logt, logtc, logtk, dlogm_height, logm0, logt0
        )

        logmpeak_jax = halo_mass_vs_time(
            time, logm0, t0=t0, logtc=logtc, logtk=logtk, dlogm_height=dlogm_height,
        )
        assert np.allclose(logmpeak_np, logmpeak_jax, rtol=0.001)


def test_halo_dmdt_vs_time_integrates_to_halo_mass_vs_time():
    logtk, dlogm_height = 3, 5
    logt0 = 1.14
    t0 = 10 ** logt0
    logtc = 0
    time = np.linspace(0.1, t0, 200)

    for logm0 in np.linspace(10, 15, 25):
        logmpeak_jax = halo_mass_vs_time(
            time, logm0, t0=t0, logtc=logtc, logtk=logtk, dlogm_height=dlogm_height,
        )
        dmhdt_jax = halo_dmdt_vs_time(
            time, logm0, t0=t0, logtc=logtc, logtk=logtk, dlogm_height=dlogm_height,
        )
        assert logmpeak_jax.shape == dmhdt_jax.shape

        integrated_logmh = np.log10(trapz(dmhdt_jax, x=time))
        assert np.allclose(integrated_logmh, logm0)
