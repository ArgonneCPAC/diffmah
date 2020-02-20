"""
"""
import numpy as np
from ..main_sequence import main_sequence_sfr_median_halo_growth
from ..main_sequence import main_sequence_sfr_vs_mpeak_and_redshift


REDSHIFT = np.array((0.0, 2, 4, 6, 8, 10)).astype("f4")
COSMIC_TIME = np.array((13.8, 3.3, 1.5, 0.9, 0.6, 0.47)).astype("f4")


def test_main_sequence_median_growth_is_non_negative():
    sfh = main_sequence_sfr_median_halo_growth(12, REDSHIFT, COSMIC_TIME)
    assert np.all(sfh > 0)


def test_main_sequence_sfr_vs_mpeak_and_redshift():
    mpeak = np.linspace(10, 15, 100)
    redshift = np.linspace(0, 10, 100)
    sfh = main_sequence_sfr_vs_mpeak_and_redshift(mpeak, redshift)
