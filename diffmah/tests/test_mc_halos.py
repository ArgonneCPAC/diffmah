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
    dmhdt, log_mah, early, late, x0, mah_type = res
    assert dmhdt.shape == (n_halos, n_times)
    assert log_mah.shape == (n_halos, n_times)
    assert np.allclose(log_mah[:, -1], logmh)


def test_mc_halo_assembly_early_late_options():
    n_halos, n_times = int(5e4), 20
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]

    for logmh in (11, 13, 15):
        res = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="early")
        dmhdt_early, log_mah_early, early_early, late_early, x0_early, mah_type = res
        assert dmhdt_early.shape == (n_halos, n_times)
        assert log_mah_early.shape == (n_halos, n_times)
        assert np.allclose(log_mah_early[:, -1], logmh)
        assert np.allclose(mah_type, 0)

        res = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="late")
        dmhdt_late, log_mah_late, early_late, late_late, x0_late, mah_type = res
        assert dmhdt_late.shape == (n_halos, n_times)
        assert log_mah_late.shape == (n_halos, n_times)
        assert np.allclose(log_mah_late[:, -1], logmh)
        assert np.allclose(mah_type, 1)

        assert np.mean(x0_early) < np.mean(x0_late)

        res = mc_halo_population(tarr, t0, logmh, n_halos)
        dmhdt, log_mah, early, late, x0, mah_type = res

        mean_dmhdt_early = np.mean(dmhdt_early, axis=0)
        mean_dmhdt_late = np.mean(dmhdt_late, axis=0)
        mean_dmhdt_early2 = np.mean(dmhdt[mah_type == 0], axis=0)
        mean_dmhdt_late2 = np.mean(dmhdt[mah_type == 1], axis=0)
        assert np.allclose(mean_dmhdt_early, mean_dmhdt_early2, rtol=0.1)
        assert np.allclose(mean_dmhdt_late, mean_dmhdt_late2, rtol=0.1)

        mean_log_mah_early = np.mean(log_mah_early, axis=0)
        mean_log_mah_late = np.mean(log_mah_late, axis=0)
        mean_log_mah_early2 = np.mean(log_mah[mah_type == 0], axis=0)
        mean_log_mah_late2 = np.mean(log_mah[mah_type == 1], axis=0)
        assert np.allclose(mean_log_mah_early, mean_log_mah_early2, rtol=0.1)
        assert np.allclose(mean_log_mah_late, mean_log_mah_late2, rtol=0.1)
