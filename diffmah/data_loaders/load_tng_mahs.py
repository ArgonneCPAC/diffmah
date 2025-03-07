"""Module loads mass assembly history data for diffmah
"""

import numpy as np
import os
import warnings

BEBOP_TNG = "/lcrc/project/halotools/alarcon/data/"


def load_tng_data(data_drn=BEBOP_TNG):
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))
    fn = os.path.join(data_drn, "tng_diffmah.npy")
    _halos = np.load(fn)
    mahs = _halos["mpeakh"]
    mahs = np.maximum.accumulate(mahs, axis=1)

    return tng_t, mahs
