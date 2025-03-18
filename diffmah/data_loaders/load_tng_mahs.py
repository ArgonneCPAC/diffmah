"""Module loads mass assembly history data for diffmah
"""

import numpy as np
import os

BEBOP_TNG = "/lcrc/project/halotools/alarcon/data/"


def load_tng_data(data_drn=BEBOP_TNG):
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))
    fn = os.path.join(data_drn, "tng_diffmah.npy")
    _halos = np.load(fn)
    log_mahs = _halos["mpeakh"]
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)
    mahs = np.where(log_mahs == 0.0, 0.0, 10**log_mahs)
    assert mahs.max() > 1e5
    return tng_t, mahs
