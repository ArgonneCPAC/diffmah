"""
"""
from ..in_situ_history import in_situ_galaxy_halo_history


def test_get_model_param_dictionaries():
    X = in_situ_galaxy_halo_history(12)
    zarr, tarr, mah, dmd = X[:4]
    sfr_ms_history, sfr_q_history = X[4:6]
    mstar_ms_history, mstar_q_history = X[6:]
