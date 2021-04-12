"""
"""
import numpy as np


def measure_fitting_residuals(tarr, log_mah_sim, log_mah_fit, lgm_min, dlgm_min):
    assert tarr.size == log_mah_sim.shape[1], "shape mismatch between log_mah and tarr"

    nh, nt = log_mah_sim.shape
    avg_error = np.zeros(nt)
    std_error = np.zeros(nt)

    gen = zip(range(nt), tarr)
    for i, t in gen:
        log_mahs_at_t = log_mah_sim[:, i]
        log_mah_errs_at_t = log_mah_fit[:, i] - log_mahs_at_t

        low_mass_msk = log_mahs_at_t > lgm_min
        dlgm_msk = log_mahs_at_t > log_mah_sim[:, -1] - dlgm_min
        msk = low_mass_msk & dlgm_msk

        if msk.sum() >= 10:
            avg_error[i] = np.mean(log_mah_errs_at_t[msk])
            std_error[i] = np.std(log_mah_errs_at_t[msk])
    return avg_error, std_error
