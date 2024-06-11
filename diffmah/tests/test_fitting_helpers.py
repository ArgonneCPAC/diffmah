"""
"""

import numpy as np

from .. import fitting_helpers as fithelp
from ..diffmah_kernels import MAH_K, _rolling_plaw_vs_logt


def test_get_loss_data():
    t_sim = np.linspace(0.1, 13.8, 100)
    logt = np.log10(t_sim)
    logt0 = logt[-1]
    lgm_min = 7.0
    logmp = 13.0
    logtc = 0.1
    early, late = 2.0, 0.5
    log_mah_sim = _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, MAH_K, early, late)

    u_p_init, loss_data = fithelp.get_loss_data(t_sim, log_mah_sim, lgm_min)
    t_target, log_mah_target, t_peak, logt0_out = loss_data
    assert np.all(np.isfinite(u_p_init))
    assert np.allclose(logt0, logt0_out)
    assert np.allclose(t_peak, t_sim[-1])
