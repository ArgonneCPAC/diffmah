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


def test_reasonable_fiducial_values_of_get_mean_galaxy_history_milky_way():
    """Enforce that the get_mean_galaxy_history function returns
    results that are reasonably consistent with hard-coded UniverseMachine.
    """
    tobs = np.linspace(1, 13.85, 10)
    log_ssfr_um_logm12 = np.array(
        [
            -8.48368008,
            -8.98833771,
            -9.09281158,
            -9.18228601,
            -9.45236933,
            -9.76191952,
            -10.01235123,
            -10.18335045,
            -10.37827302,
            -10.5594088,
        ]
    )
    log_sm_um_logm12 = np.array(
        [
            7.87270177,
            8.81792753,
            9.38892252,
            9.94781192,
            10.25347201,
            10.40364212,
            10.48279124,
            10.53272313,
            10.56485557,
            10.58462598,
        ]
    )
    log_sfr_pred_logm12, log_sm_pred_logm12 = get_mean_galaxy_history(12, tobs)
    log_ssfr_pred_logm12 = log_sfr_pred_logm12 - log_sm_pred_logm12

    diff_logssfr_logm12 = log_ssfr_pred_logm12 - log_ssfr_um_logm12
    diff_logsm_logm12 = log_sm_pred_logm12 - log_sm_um_logm12
    n = diff_logssfr_logm12.size
    loss_ssfr = np.sum(diff_logssfr_logm12 * diff_logssfr_logm12) / n
    loss_sm = np.sum(diff_logsm_logm12 * diff_logsm_logm12) / n
    assert np.log10(loss_sm) < -2
    assert np.log10(loss_ssfr) < -2


def test_reasonable_fiducial_values_of_get_mean_galaxy_history_groups():
    """Enforce that the get_mean_galaxy_history function returns
    results that are reasonably consistent with hard-coded UniverseMachine.
    """
    tobs = np.linspace(1, 13.85, 10)

    log_ssfr_um_logm13 = np.array(
        [
            -8.43244273,
            -8.87366801,
            -9.33136302,
            -9.93573076,
            -10.26354487,
            -10.38942505,
            -10.59986553,
            -10.64110237,
            -10.66155861,
            -10.78584939,
        ]
    )
    log_sm_um_logm13 = np.array(
        [
            8.82167812,
            10.25849786,
            10.79993315,
            10.93721088,
            10.98577248,
            11.01543131,
            11.03427237,
            11.04900105,
            11.06240706,
            11.07288288,
        ]
    )

    log_sfr_pred_logm13, log_sm_pred_logm13 = get_mean_galaxy_history(13, tobs)
    log_ssfr_pred_logm13 = log_sfr_pred_logm13 - log_sm_pred_logm13

    diff_logssfr_logm13 = log_ssfr_pred_logm13 - log_ssfr_um_logm13
    diff_logsm_logm13 = log_sm_pred_logm13 - log_sm_um_logm13
    n = diff_logssfr_logm13.size
    loss_ssfr = np.sum(diff_logssfr_logm13 * diff_logssfr_logm13) / n
    loss_sm = np.sum(diff_logsm_logm13 * diff_logsm_logm13) / n
    assert np.log10(loss_sm) < -2
    assert np.log10(loss_ssfr) < -2
