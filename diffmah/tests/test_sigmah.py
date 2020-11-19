"""
"""
import numpy as np
from ..sigmah import individual_halo_assembly_history


def test_individual_mah_evaluates():
    nt, nm = 50, 20
    tarr = np.linspace(0.5, 14, nt)
    lgmarr = np.linspace(10, 15, nm)
    log_mah, log_dmhdt = individual_halo_assembly_history(tarr, lgmarr)
    assert log_mah.shape == log_dmhdt.shape == (nm, nt)
