"""
"""
import os
import numpy as np
from ..individual_halo_assembly import _calc_halo_history, DEFAULT_MAH_PARAMS
from ..individual_halo_assembly import _power_law_index_vs_logt
from ..rockstar_pdf_model import _get_cov_params_early, _get_cov_params_late
from ..rockstar_pdf_model import _get_mean_mah_params_early, _get_mean_mah_params_late


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_calc_halo_history_evaluates():
    tarr = np.linspace(0.1, 14, 500)
    logt = np.log10(tarr)
    logtmp = logt[-1]
    logmp = 12.0
    dmhdt, log_mah = _calc_halo_history(
        logt, logtmp, logmp, *DEFAULT_MAH_PARAMS.values()
    )


def test_rolling_index_agrees_with_hard_coded_expectation():
    lgmp_test = 12.5

    k = DEFAULT_MAH_PARAMS["mah_k"]
    logt_bn = "logt_testing_array_logmp_{0:.1f}.dat".format(lgmp_test)
    logt = np.loadtxt(os.path.join(DDRN, logt_bn))
    logt0 = logt[-1]

    lge_e, lgl_e, lgtc_e = _get_mean_mah_params_early(lgmp_test)
    lge_l, lgl_l, lgtc_l = _get_mean_mah_params_late(lgmp_test)
    _z1 = np.zeros(1)
    cov_p_e = _get_cov_params_early(_z1 + lgmp_test)
    cov_p_l = _get_cov_params_late(_z1 + lgmp_test)
    # rolling_index = _rolling_plaw_vs_logt(logt, logt0, logmp, logtc, k, early, late)
