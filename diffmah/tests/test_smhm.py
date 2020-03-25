"""
"""
import numpy as np
from astropy.cosmology import Planck15
from ..smhm import mstar_vs_mhalo_at_zobs
from ..sigmoid_mah import logm0_from_logm_at_logt


EPSILON = 1e-4


def test_mstar_vs_mhalo_at_zobs_has_reasonable_quenching_behavior():
    """Quenching cannot increase stellar mass."""
    logmhalo = np.linspace(10, 16, 30)
    for zobs in (0, 0.5, 1, 2, 5):
        tobs = Planck15.age(zobs).value
        logsm_ms, logsm_med, logsm_q = mstar_vs_mhalo_at_zobs(zobs, tobs, logmhalo)
        assert np.all(logsm_ms >= logsm_med - EPSILON)
        assert np.all(logsm_med >= logsm_q - EPSILON)


def test_mstar_vs_mhalo_at_zobs_has_reasonable_mass_dependence():
    """logM* should increase monotonically with logMh for unquenched galaxies."""
    logmhalo = np.linspace(10, 15, 20)
    for zobs in (0, 0.5, 1, 2, 5):
        tobs = Planck15.age(zobs).value
        logsm_ms, logsm_med, logsm_q = mstar_vs_mhalo_at_zobs(zobs, tobs, logmhalo)
        assert np.all(np.diff(logsm_ms) > 0)


def test_mstar_vs_mhalo_at_zobs_has_reasonable_z_dependence():
    """logM* should decrease monotonically with redshift."""
    logmh = 12
    zray = np.array((0, 0.5, 1, 2, 5))
    logsm = np.zeros_like(zray)
    for i, zobs in enumerate(zray):
        tobs = Planck15.age(zobs).value
        _x, __, __ = mstar_vs_mhalo_at_zobs(zobs, tobs, logmh)
        logsm[i] = _x
    assert np.all(np.diff(logsm) > 0)


def test_mstar_vs_mhalo_has_reasonable_mah_dependence():
    """
    """
    zobs = 1
    tobs = Planck15.age(zobs).value
    logmh_at_zobs = 12
    mstar1, __, __ = mstar_vs_mhalo_at_zobs(zobs, tobs, logmh_at_zobs, logtc=-0.2)
    mstar2, __, __ = mstar_vs_mhalo_at_zobs(zobs, tobs, logmh_at_zobs, logtc=0.2)

    assert mstar1[0] != mstar2[0]
