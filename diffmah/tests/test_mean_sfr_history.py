"""
"""
import numpy as np
from ..mean_sfr_history import _mean_log_mstar_history_jax_kern
from ..mean_sfr_history import _mean_log_sfr_history_jax_kern
from ..quenching_history import MEAN_Q_PARAMS
from ..main_sequence_sfr_eff import MEAN_SFR_MS_PARAMS
from ..halo_assembly import MEAN_MAH_PARAMS
from ..mean_sfr_history import get_mean_galaxy_history


def test_mean_galaxy_history():
    tarr = np.linspace(0.1, 14, 500)
    for logm0 in range(10, 16):
        log_sfr, log_sm = get_mean_galaxy_history(logm0, tarr)
        assert log_sfr.size == log_sm.size == tarr.size


def test_mean_log_sfr_history():
    mah_params = np.array(list(MEAN_MAH_PARAMS.values()))
    mean_sfr_eff_params = np.array(list(MEAN_SFR_MS_PARAMS.values()))
    q_params = np.array(list(MEAN_Q_PARAMS.values()))
    logm0 = 12
    logt = np.linspace(0, 1.141, 50)
    indx_t0 = -1
    log_sfrh = _mean_log_sfr_history_jax_kern(
        mah_params, mean_sfr_eff_params, q_params, logm0, logt, indx_t0
    )
    assert np.all(np.isfinite(log_sfrh))


def test_mean_log_mstar_history():
    mah_params = np.array(list(MEAN_MAH_PARAMS.values()))
    mean_sfr_eff_params = np.array(list(MEAN_SFR_MS_PARAMS.values()))
    q_params = np.array(list(MEAN_Q_PARAMS.values()))
    logm0 = 12
    tarr = np.linspace(0.1, 13.85, 50)
    logt_table = np.log10(tarr)
    indx_t0 = -1
    dt = np.diff(tarr)[-1]
    indx_pred = np.array((10, 20, 30)).astype("i4")
    log_sfrh, log_smh = _mean_log_mstar_history_jax_kern(
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
