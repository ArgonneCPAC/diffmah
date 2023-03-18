"""
"""
import numpy as np
from ..monte_carlo_halo_population import mc_halo_population
from ..monte_carlo_halo_population import mc_halo_population2


def test_diff_nondiff_mc_halopop_agree():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0 + np.zeros(n_halos)
    mc_halopop = mc_halo_population(tarr, t0, logmh)
    mc_halopop2 = mc_halo_population2(tarr, t0, logmh)

    assert np.allclose(mc_halopop.log_mah, mc_halopop2.log_mah, rtol=1e-4)
