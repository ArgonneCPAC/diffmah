"""
"""
import numpy as np
from ..halo_population_assembly import _get_bimodal_halo_history


def test_get_average_halo_histories():
    """Verify that the _get_average_halo_histories returns reasonable arrays."""
    n_early, n_late, n_x0 = 10, 15, 20
    lge_min, lge_max = -1.5, 1.5
    lgl_min, lgl_max = -2, 1
    x0_min, x0_max = -1.0, 1
    lge_arr = np.linspace(lge_min, lge_max, n_early)
    lgl_arr = np.linspace(lgl_min, lgl_max, n_late)
    x0_arr = np.linspace(x0_min, x0_max, n_x0)
    tarr = np.linspace(1, 13.8, 25)
    lgt_arr = np.log10(tarr)
    lgmp_arr = np.array((11.25, 11.75, 12, 12.5, 13, 13.5, 14, 14.5))
    _res = _get_bimodal_halo_history(lgt_arr, lgmp_arr, lge_arr, lgl_arr, x0_arr)
    mean_dmhdt, mean_mah, variance_dmhdt, variance_mah = _res
    mean_log_mahs = np.log10(mean_mah)

    #  Average halo MAHs should agree at t=today
    assert np.allclose(mean_log_mahs[:, -1], lgmp_arr, atol=0.01)

    # Average halo MAHs should monotonically increase
    assert np.all(np.diff(mean_log_mahs, axis=1) > 0)

    # Average halo accretion rates should monotonically increase with present-day mass
    assert np.all(np.diff(mean_dmhdt[:, -1]) > 0)
