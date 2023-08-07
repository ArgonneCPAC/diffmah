"""
"""
import os

import numpy as np

GALPROPS = ["halo_id", "mpeak_history_main_prog", "sfr_history_main_prog"]
LOGMH_MIN = 7.0


def load_smdpl_histories(drn, subvolumes, galprops=GALPROPS, lgmh_min=LOGMH_MIN):
    from umachine_pyio.load_mock import load_mock_from_binaries

    mock = load_mock_from_binaries(subvolumes, drn, galprops)
    tarr = np.loadtxt(os.path.join(drn, "smdpl_cosmic_time.txt"))

    mh_min = 10**lgmh_min
    msk = mock["mpeak_history_main_prog"] < mh_min
    clipped_mahs = np.where(msk, mh_min, mock["mpeak_history_main_prog"])
    log_mahs = np.log10(clipped_mahs)
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)
    mock.remove_column("mpeak_history_main_prog")
    mock["log_mah"] = log_mahs

    return mock, tarr, lgmh_min
