"""
"""
import numpy as np
from .in_situ_history import in_situ_galaxy_halo_history


def quenched_fraction_at_zobs(logm0, zobs, **kwargs):
    """Integrate star formation history to calculate M* at zobs."""
    _X = in_situ_galaxy_halo_history(logm0, **kwargs)
    zarr, tarr, mah, dmhdt = _X[:4]
    sfrh_ms, sfrh_q, smh_ms, smh_q = _X[4:]

    return qfrac
