"""
"""
import numpy as np
from ..epsilon_sfr_emerge import sfr_efficiency_function, DEFAULT_PARAMS


def test_function_changes_with_redshift():
    npts = int(1e4)
    mhalo_at_z = np.logspace(10, 15, npts)
    epsilon_at_z0 = sfr_efficiency_function(mhalo_at_z, 0.0)
    epsilon_at_z1 = sfr_efficiency_function(mhalo_at_z, 1.0)
    assert not np.allclose(epsilon_at_z0, epsilon_at_z1)


def test_function_changes_with_all_params():
    npts = int(1e4)
    mhalo_at_z = np.logspace(10, 15, npts)
    ztest = 0.5
    epsilon_at_ztest = sfr_efficiency_function(mhalo_at_z, ztest)
    for key, default in DEFAULT_PARAMS.items():
        alt_val = 1.5 * default
        d = {key: alt_val}
        epsilon_at_ztest_alt = sfr_efficiency_function(mhalo_at_z, ztest, **d)
        assert not np.allclose(epsilon_at_ztest, epsilon_at_ztest_alt)
