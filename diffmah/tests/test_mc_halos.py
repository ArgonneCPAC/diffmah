"""
"""
import numpy as np
from ..monte_carlo_halo_population import mc_halo_population


def test_mc_halo_assembly_returns_reasonable_results():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0
    res = mc_halo_population(tarr, t0, logmh, n_halos)
    dmhdt, log_mah, early, late, x0 = res
    assert dmhdt.shape == (n_halos, n_times)
    assert log_mah.shape == (n_halos, n_times)
    assert np.allclose(log_mah[:, -1], logmh)


def test_mc_halo_assembly_early_late_options():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0

    for logmh in (11, 12, 13, 14, 15):
        res = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="early")
        dmhdt_early, log_mah_early, early_early, late_early, x0_early = res
        assert dmhdt_early.shape == (n_halos, n_times)
        assert log_mah_early.shape == (n_halos, n_times)
        assert np.allclose(log_mah_early[:, -1], logmh)

        res = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="late")
        dmhdt_late, log_mah_late, early_late, late_late, x0_late = res
        assert dmhdt_late.shape == (n_halos, n_times)
        assert log_mah_late.shape == (n_halos, n_times)
        assert np.allclose(log_mah_late[:, -1], logmh)

        assert np.mean(x0_early) < np.mean(x0_late)
