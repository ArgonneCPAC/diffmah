"""
"""
import numpy as np
from ..star_formation_history import _mean_log_sfr_history, _mean_log_mstar_history
from ..quenching_history import MEDIAN_HISTORY_PARAMS as MEAN_Q_PARAMS
from ..sfr_efficiency import MEDIAN_SFR_MS_PARAMS
from ..halo_assembly import MEAN_MAH_PARAMS


def test_mean_log_sfr_history():
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


def test_mean_log_mstar_history():
    mah_params = np.array(list(MEAN_MAH_PARAMS.values()))
    mean_sfr_eff_params = np.array(list(MEDIAN_SFR_MS_PARAMS.values()))
    q_params = np.array(list(MEAN_Q_PARAMS.values()))
    logm0 = 12
    tarr = np.linspace(0.1, 13.85, 50)
    logt_table = np.log10(tarr)
    indx_t0 = -1
    dt = np.diff(tarr)[-1]
    indx_pred = np.array((10, 20, 30)).astype("i4")
    log_sfrh, log_smh = _mean_log_mstar_history(
        mah_params,
        mean_sfr_eff_params,
        q_params,
        logm0,
        logt_table,
        indx_t0,
        dt,
        indx_pred,
    )
    assert log_sfrh.size == indx_pred.size == log_smh.size
