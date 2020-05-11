"""
"""
import numpy as np
from ..star_formation_history import _mean_log_sfr_history
from ..quenching_history import MEDIAN_HISTORY_PARAMS as MEAN_Q_PARAMS
from ..sfr_efficiency import MEDIAN_SFR_MS_PARAMS
from ..halo_assembly import MEAN_MAH_PARAMS


def test1():
    mah_params = np.array(list(MEAN_MAH_PARAMS.values()))
    mean_sfr_eff_params = np.array(list(MEDIAN_SFR_MS_PARAMS.values()))
    q_params = np.array(list(MEAN_Q_PARAMS.values()))
    logm0 = 12
    logt = np.linspace(0, 1.141, 50)
    indx_t0 = -1
    log_sfrh = _mean_log_sfr_history(
        mah_params, mean_sfr_eff_params, q_params, logm0, logt, indx_t0
    )
    assert np.all(np.isfinite(log_sfrh))
