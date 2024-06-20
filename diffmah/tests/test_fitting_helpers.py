"""
"""

import numpy as np

from .. import fitting_helpers as fithelp
from ..bfgs_wrapper import diffmah_fitter
from ..diffmah_kernels import (
    DiffmahParams,
    DiffmahUParams,
    _rolling_plaw_vs_logt,
    get_bounded_mah_params,
)


def test_fitting_helpers_integration():
    t_sim = np.linspace(0.1, 13.8, 2000)
    logt = np.log10(t_sim)
    logt0 = logt[-1]
    lgm_min = 7.0
    logm0 = 13.0
    logtc = 0.4
    early, late = 0.8, 0.15
    p_true = DiffmahParams(logm0, logtc, early, late)
    log_mah_sim = _rolling_plaw_vs_logt(logt, logt0, logm0, logtc, early, late)

    u_p_init, loss_data = fithelp.get_loss_data(t_sim, log_mah_sim, lgm_min)
    t_target, log_mah_target, t_peak, logt0_out = loss_data
    assert np.all(np.isfinite(u_p_init))
    assert np.allclose(logt0, logt0_out)
    assert np.allclose(t_peak, t_sim[-1])

    msk = t_sim >= t_target.min()
    assert np.allclose(log_mah_target, log_mah_sim[msk])

    _res = diffmah_fitter(fithelp.loss_and_grads_kern, u_p_init, loss_data)
    u_p_best, loss_best, fit_terminates, code_used = _res

    ATOL = 0.1

    assert fit_terminates
    assert code_used == 0
    assert loss_best < 0.001
    p_best_inferred = get_bounded_mah_params(DiffmahUParams(*u_p_best))
    assert np.allclose(p_best_inferred, p_true, rtol=ATOL)
    npts_mah = log_mah_target.size

    root_indx = 123
    outline = fithelp.get_outline(
        root_indx, loss_data, u_p_best, loss_best, npts_mah, code_used
    )
    outdata = [float(x) for x in outline.strip().split()]
    header_data = fithelp.HEADER[1:].strip().split()
    assert len(header_data) == len(outdata)
    assert np.allclose(outdata[1], logm0, rtol=ATOL)
    assert np.allclose(outdata[2], logtc, rtol=ATOL)
    assert np.allclose(outdata[3], early, rtol=ATOL)
    assert np.allclose(outdata[4], late, rtol=ATOL)
    assert np.allclose(outdata[5], t_peak, rtol=ATOL)
