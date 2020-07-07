"""
"""
import numpy as np
from ..individual_sfr_history import individual_sfr_history

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
    log_sfr_table, log_sm_table = individual_sfr_history(12, t_table)
    assert np.all(np.isfinite(log_sfr_table))
    assert np.all(np.isfinite(log_sm_table))

    log_sm_pred = np.interp(TOBS, t_table, log_sm_table)
    diff = log_sm_pred - LOG_SMH_UM_LOGM12
    mse = np.sum(diff * diff) / diff.size
    assert np.log10(mse) < -1.0
