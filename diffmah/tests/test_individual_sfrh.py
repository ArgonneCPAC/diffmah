"""
"""
import numpy as np
from ..individual_sfr_history import individual_sfr_history
from ..individual_sfr_history import predict_in_situ_history_collection
from ..individual_sfr_history import DEFAULT_SFRH_PARAMS
from ..halo_assembly import DEFAULT_MAH_PARAMS, TODAY


TOBS = np.linspace(1, 13.8, 10)
LOG_SMH_UM_LOGM12 = np.array(
    [
        7.87270177,
        8.81537823,
        9.38485967,
        9.94297962,
        10.25004252,
        10.40156255,
        10.48140772,
        10.53163869,
        10.56405419,
        10.58425636,
    ]
)


def test_individual_sfr_history_agrees_with_umachine_milky_way_halos():
    """Enforce reasonable agreement with a hard-coded tabulation of
    the mean SFR history of UniverseMachine centrals with logmp=12.
    These should agree because the values of DEFAULT_SFRH_PARAMS
    have been set according to the calibrated value of the global model
    evaluated at logMhalo=12.
    """
    t_table = np.linspace(0.1, 14, 500)
    log_sfr_table, log_sm_table = individual_sfr_history(t_table, 12)
    assert np.all(np.isfinite(log_sfr_table))
    assert np.all(np.isfinite(log_sm_table))

    log_sm_pred = np.interp(TOBS, t_table, log_sm_table)
    diff = log_sm_pred - LOG_SMH_UM_LOGM12
    mse = np.sum(diff * diff) / diff.size
    assert np.log10(mse) < -1.0


def test_predict_in_situ_history_collection_returns_correct_fstar():
    nh, nt = 15, 1500
    t = np.linspace(0.5, 14, nt)
    mah_params = np.zeros((nh, 6)).astype("f4")
    sfr_params = np.zeros((nh, len(DEFAULT_SFRH_PARAMS))).astype("f4")
    for ih in range(nh):
        mah_params[ih, 0] = TODAY
        mah_params[ih, 1] = 12.0
        mah_params[ih, 2] = DEFAULT_MAH_PARAMS["dmhdt_x0"]
        mah_params[ih, 3] = DEFAULT_MAH_PARAMS["dmhdt_k"]
        mah_params[ih, 4] = DEFAULT_MAH_PARAMS["dmhdt_early_index"]
        mah_params[ih, 5] = DEFAULT_MAH_PARAMS["dmhdt_late_index"]
        for ip, val in enumerate(DEFAULT_SFRH_PARAMS.values()):
            sfr_params[ih, ip] = val

    _x0 = predict_in_situ_history_collection(mah_params, sfr_params, t)
    _x1 = predict_in_situ_history_collection(
        mah_params, sfr_params, t, fstar_timescales=0.25
    )
    _x2 = predict_in_situ_history_collection(
        mah_params, sfr_params, t, fstar_timescales=(0.25, 1.0)
    )
    assert np.allclose(_x0[0], _x1[0])
    assert np.allclose(_x0[0], _x2[0])
    assert np.allclose(_x0[1], _x1[1])
    assert np.allclose(_x0[1], _x2[1])
    assert np.allclose(_x1[2], _x2[2])

    fs0p25 = _x2[2][0, :]
    fs1p0 = _x2[3][0, :]
    assert not np.allclose(fs0p25[-1], fs1p0[-1], rtol=0.01)

    t0_lag_0p25 = t.max() - 0.25
    t0_lag_1p0 = t.max() - 1
    sm_at_t0_lag_0p25 = 10 ** np.interp(
        np.log10(t0_lag_0p25), np.log10(t), _x0[0][0, :]
    )
    sm_at_t0_lag_1p0 = 10 ** np.interp(np.log10(t0_lag_1p0), np.log10(t), _x0[0][0, :])
    sm_at_t0 = 10 ** _x0[0][0, -1]

    correct_fstar_0p25_today = 1 - sm_at_t0_lag_0p25 / sm_at_t0
    correct_fstar_1p0_today = 1 - sm_at_t0_lag_1p0 / sm_at_t0

    assert np.allclose(correct_fstar_0p25_today, fs0p25[-1], rtol=0.001)
    assert np.allclose(correct_fstar_1p0_today, fs1p0[-1], rtol=0.001)
