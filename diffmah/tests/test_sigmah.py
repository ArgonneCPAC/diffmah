"""
"""
import numpy as np
from ..sigmah import individual_halo_assembly_history
from ..sigmah import _get_bounded_params, _get_unbounded_params


def test_bounded_params_inverts():
    early_index, late_index = 3.0, 1.0
    log_early_index, u_dy = _get_unbounded_params(early_index, late_index)
    assert np.allclose(log_early_index, np.log10(early_index))
    early_index_inv, late_index_inv = _get_bounded_params(log_early_index, u_dy)
    assert np.allclose(early_index_inv, early_index)
    assert np.allclose(late_index_inv, late_index)


def test_individual_mah_evaluates():
    nt, nm = 50, 20
    tarr = np.linspace(0.5, 14, nt)
    lgmarr = np.linspace(10, 15, nm)
    log_mah, log_dmhdt = individual_halo_assembly_history(tarr, lgmarr)
    assert log_mah.shape == log_dmhdt.shape == (nm, nt)
