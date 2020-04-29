"""
"""
import numpy as np
from ..halo_mah import halo_mass_assembly_history


def test_halo_mah_evaluates_reasonably_with_default_args():
    """
    """
    npts = 50
    for logm0 in (11, 12, 13, 14, 15):
        for t0 in (13.5, 14):
            cosmic_time = np.linspace(0.1, t0, npts)
            logmah, log_dmhdt = halo_mass_assembly_history(logm0, cosmic_time, t0=t0)
            assert logmah.size == npts == log_dmhdt.size
            assert np.allclose(logmah[-1], logm0, atol=0.01)
