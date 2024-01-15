"""
"""
import os

import numpy as np
from jax import numpy as jnp

from ..defaults import DEFAULT_MAH_PARAMS, MAH_K, DiffmahParams
from ..individual_halo_assembly import (
    _calc_halo_history,
    _calc_halo_history_scalar,
    _get_early_late,
    _power_law_index_vs_logt,
    mah_halopop,
    mah_singlehalo,
)
from ..rockstar_pdf_model import _get_mean_mah_params_early, _get_mean_mah_params_late

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_calc_halo_history_evaluates():
    tarr = np.linspace(0.1, 14, 500)
    logt = np.log10(tarr)
    logtmp = logt[-1]
    lgmp, lgtc, early, late = DEFAULT_MAH_PARAMS
    dmhdt, log_mah = _calc_halo_history(logt, logtmp, lgmp, lgtc, MAH_K, early, late)
    assert np.all(np.isfinite(dmhdt))
    assert np.all(np.isfinite(log_mah))


def test_rolling_index_agrees_with_hard_coded_expectation():
    lgmp_test = 12.5

    logt_bn = "logt_testing_array.dat"
    logt = np.loadtxt(os.path.join(DDRN, logt_bn))
    logt0 = logt[-1]

    ue_e, ul_e, lgtc_e = _get_mean_mah_params_early(lgmp_test)
    ue_l, ul_l, lgtc_l = _get_mean_mah_params_late(lgmp_test)

    early_e, late_e = _get_early_late(ue_e, ul_e)
    early_l, late_l = _get_early_late(ue_l, ul_l)

    indx_e_bn = "rolling_plaw_index_vs_time_rockstar_default_logmp_{0:.1f}_early.dat"
    index_early_correct = np.loadtxt(os.path.join(DDRN, indx_e_bn.format(lgmp_test)))
    index_early = _power_law_index_vs_logt(logt, lgtc_e, MAH_K, early_e, late_e)
    assert np.allclose(index_early_correct, index_early, rtol=0.01)

    indx_l_bn = "rolling_plaw_index_vs_time_rockstar_default_logmp_{0:.1f}_late.dat"
    index_late_correct = np.loadtxt(os.path.join(DDRN, indx_l_bn.format(lgmp_test)))
    index_late = _power_law_index_vs_logt(logt, lgtc_l, MAH_K, early_l, late_l)
    assert np.allclose(index_late_correct, index_late, rtol=0.01)

    dmhdt_e, log_mah_e = _calc_halo_history(
        logt, logt0, lgmp_test, lgtc_e, MAH_K, early_e, late_e
    )
    dmhdt_l, log_mah_l = _calc_halo_history(
        logt, logt0, lgmp_test, lgtc_l, MAH_K, early_l, late_l
    )

    log_mah_e_bn = "log_mah_vs_time_rockstar_default_logmp_{0:.1f}_early.dat"
    log_mah_e_correct = np.loadtxt(os.path.join(DDRN, log_mah_e_bn.format(lgmp_test)))
    assert np.allclose(log_mah_e_correct, log_mah_e, atol=0.01)

    log_mah_l_bn = "log_mah_vs_time_rockstar_default_logmp_{0:.1f}_late.dat"
    log_mah_l_correct = np.loadtxt(os.path.join(DDRN, log_mah_l_bn.format(lgmp_test)))
    assert np.allclose(log_mah_l_correct, log_mah_l, atol=0.01)


def test_calc_halo_history_scalar_agrees_with_vmap():
    tarr = np.linspace(0.1, 14, 15)
    logt = np.log10(tarr)
    logtmp = logt[-1]
    lgmp, lgtc, early, late = DEFAULT_MAH_PARAMS
    dmhdt, log_mah = _calc_halo_history(logt, logtmp, lgmp, lgtc, MAH_K, early, late)

    for i, t in enumerate(tarr):
        lgt_i = jnp.log10(t)
        res = _calc_halo_history_scalar(lgt_i, logtmp, lgmp, lgtc, MAH_K, early, late)
        dmhdt_i, log_mah_i = res
        assert np.allclose(dmhdt[i], dmhdt_i)
        assert np.allclose(log_mah[i], log_mah_i)


def test_mah_singlehalo_evaluates():
    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)
    dmhdt, log_mah = mah_singlehalo(DEFAULT_MAH_PARAMS, tarr)
    assert dmhdt.shape == tarr.shape
    assert log_mah.shape == dmhdt.shape
    assert log_mah[-1] == DEFAULT_MAH_PARAMS.logmp


def test_mah_halopop_evaluates():
    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)

    ngals = 150
    zz = np.zeros(ngals)
    mah_params_halopop = DiffmahParams(*[zz + p for p in DEFAULT_MAH_PARAMS])
    dmhdt, log_mah = mah_halopop(mah_params_halopop, tarr)
    assert dmhdt.shape == (ngals, nt)
    assert log_mah.shape == dmhdt.shape
    assert np.allclose(log_mah[:, -1], DEFAULT_MAH_PARAMS.logmp)
