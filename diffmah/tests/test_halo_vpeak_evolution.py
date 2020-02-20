"""
"""
import numpy as np
from ..halo_vpeak_evolution import vmax_vs_mhalo_and_redshift


def test_vmax_vs_mhalo_and_redshift_evaluates():
    """Ensure function properly evaluates."""
    npts = 100
    m0, z0 = 10**10, 1
    m0arr = np.zeros(npts) + m0
    zarr = np.zeros(npts) + z0
    vmax0 = vmax_vs_mhalo_and_redshift(m0arr, zarr)
