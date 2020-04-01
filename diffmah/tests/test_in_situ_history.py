"""
"""
import numpy as np
from ..in_situ_history import in_situ_galaxy_halo_history
from ..in_situ_smh_kernel import in_situ_mstar_at_zobs


def test_get_model_param_dictionaries():
    X = in_situ_galaxy_halo_history(12)
    zarr, tarr, mah, dmd = X[:4]
    sfr_ms_history, sfr_q_history = X[4:6]
    mstar_ms_history, mstar_q_history = X[6:]

    mstar_ms, mstar_q = in_situ_mstar_at_zobs(0, 12)
    assert np.allclose(mstar_ms_history[-1], mstar_ms, rtol=0.01)
    assert np.allclose(mstar_q_history[-1], mstar_q, rtol=0.01)
