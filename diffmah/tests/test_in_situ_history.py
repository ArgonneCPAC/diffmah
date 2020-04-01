"""
"""
import pytest
from collections import OrderedDict
import numpy as np
from ..in_situ_history import in_situ_galaxy_halo_history, _get_model_param_dictionaries
from ..sigmoid_mah import DEFAULT_MAH_PARAMS, _median_mah_sigmoid_params
from ..moster17_efficiency import DEFAULT_PARAMS as DEFAULT_SFR_PARAMS
from ..quenching_times import DEFAULT_PARAMS as DEFAULT_QTIME_PARAMS


def test_get_model_param_dictionaries():
    defaults = [OrderedDict(a=1, b=2), OrderedDict(c=3, d=4)]
    new_vals = OrderedDict(a=2)
    result = _get_model_param_dictionaries(*defaults, **new_vals)
    correct = [OrderedDict(a=2, b=2), OrderedDict(c=3, d=4)]
    assert np.all([x == y for x, y in zip(result, correct)])


def test2_get_model_param_dictionaries():
    defaults = [OrderedDict(a=1, b=2), OrderedDict(c=3, d=4)]
    new_vals = OrderedDict(c=4)
    result = _get_model_param_dictionaries(*defaults, **new_vals)
    correct = [OrderedDict(a=1, b=2), OrderedDict(c=4, d=4)]
    assert np.all([x == y for x, y in zip(result, correct)])


def test3_get_model_param_dictionaries():
    defaults = [OrderedDict(a=1, b=2), OrderedDict(c=3, d=4)]
    new_vals = OrderedDict(e=5)
    with pytest.raises(KeyError):
        _get_model_param_dictionaries(*defaults, **new_vals)


def test4_get_model_param_dictionaries():
    defaults = [OrderedDict(a=1, b=2), OrderedDict(a=3, d=4)]
    new_vals = OrderedDict(d=5)
    with pytest.raises(KeyError):
        _get_model_param_dictionaries(*defaults, **new_vals)


def test1_in_situ_galaxy_halo_history_is_monotonic_in_mass():
    """Enforce M*(Mhalo, z) is monotonic in mass
    at each redshift of the returned integrand"""
    X10 = in_situ_galaxy_halo_history(10)
    X12 = in_situ_galaxy_halo_history(12)
    X14 = in_situ_galaxy_halo_history(14)

    mstarh_logm10 = X10[-2]
    mstarh_logm12 = X12[-2]
    mstarh_logm14 = X14[-2]

    assert np.all(mstarh_logm10 < mstarh_logm12)
    assert np.all(mstarh_logm12 < mstarh_logm14)


def test_in_situ_galaxy_halo_history_is_less_for_quenched_galaxies():
    """
    """
    for logM in np.linspace(10, 15, 15):
        X = in_situ_galaxy_halo_history(logM, qtime=1)
        mstarh_logm10, mstarh_logm10_q = X[-2], X[-1]
        assert np.all(mstarh_logm10 >= mstarh_logm10_q)
        assert np.any(mstarh_logm10 > mstarh_logm10_q)


def test_in_situ_galaxy_halo_history_scales_correctly_with_mah_percentile():
    """Earlier-forming halos have greater M* today."""
    X0 = in_situ_galaxy_halo_history(12, mah_percentile=0)
    X0p5 = in_situ_galaxy_halo_history(12)
    X1 = in_situ_galaxy_halo_history(12, mah_percentile=1)

    mstarh0 = X0[-2]
    mstarh0p5 = X0p5[-2]
    mstarh1 = X1[-2]

    assert mstarh0[-1] > mstarh0p5[-1] > mstarh1[-1]
    assert np.all(mstarh0 >= mstarh0p5)
    assert np.any(mstarh0 > mstarh0p5)
    assert np.all(mstarh0p5 >= mstarh1)
    assert np.any(mstarh0p5 > mstarh1)


def test_in_situ_galaxy_halo_history_scales_correctly_with_logtc():
    """Earlier-forming halos have greater M* today."""
    X0 = in_situ_galaxy_halo_history(12, logtc=-0.5)
    X0p5 = in_situ_galaxy_halo_history(12, logtc=0)
    X1 = in_situ_galaxy_halo_history(12, logtc=0.5)
    mstarh0 = X0[-2]
    mstarh0p5 = X0p5[-2]
    mstarh1 = X1[-2]

    assert mstarh0[-1] > mstarh0p5[-1] > mstarh1[-1]
    assert np.all(mstarh0 >= mstarh0p5)
    assert np.any(mstarh0 > mstarh0p5)
    assert np.all(mstarh0p5 >= mstarh1)
    assert np.any(mstarh0p5 > mstarh1)


def test_in_situ_galaxy_halo_history_catches_bad_mah_percentile_inputs():

    with pytest.raises(ValueError):
        in_situ_galaxy_halo_history(12, mah_percentile=1, logtc=1)


def test_in_situ_galaxy_halo_history_has_sensible_qtime_behavior():
    """Changing qtime should have the expected effect on stellar mass."""
    X1 = in_situ_galaxy_halo_history(12, qtime=1)
    X10 = in_situ_galaxy_halo_history(12, qtime=10)
    mstar_ms1, mstar_q1 = X1[-2:]
    mstar_ms10, mstar_q10 = X10[-2:]

    assert np.allclose(mstar_ms1, mstar_ms10)

    assert np.all(mstar_ms1 >= mstar_q1)
    assert np.any(mstar_ms1 > mstar_q1)

    assert np.all(mstar_ms10 >= mstar_q10)
    assert np.any(mstar_ms10 > mstar_q10)

    assert np.all(mstar_q10 >= mstar_q1)
    assert np.any(mstar_q10 > mstar_q1)

    # Intended for use with qtime = 1, 10
    assert mstar_q1[-1] < mstar_ms1[-1] * 0.9
    assert mstar_q10[-1] > mstar_ms10[-1] * 0.9


def test_in_situ_galaxy_halo_history_varies_with_MAH_params():
    """Present-day Mstar should change when each MAH param is varied."""
    mah_params_to_vary = {
        key: value for key, value in DEFAULT_MAH_PARAMS.items() if "scatter" not in key
    }
    mstar_ms_fid, mstar_q_fid = in_situ_galaxy_halo_history(12)[-2:]
    for key, value in mah_params_to_vary.items():
        X = in_situ_galaxy_halo_history(12, **{key: value * 0.9 - 0.1})
        mstar_ms, mstar_q = X[-2:]
        assert np.any(mstar_ms != mstar_ms_fid)
        assert np.any(mstar_q != mstar_q_fid)


def test_in_situ_galaxy_halo_history_varies_with_SFR_efficiency_params():
    """Present-day Mstar should change when each SFR param is varied."""
    params_to_vary = {
        key: value for key, value in DEFAULT_SFR_PARAMS.items() if "scatter" not in key
    }
    mstar_ms_fid, mstar_q_fid = in_situ_galaxy_halo_history(12)[-2:]
    for key, value in params_to_vary.items():
        X = in_situ_galaxy_halo_history(12, **{key: value * 0.9 - 0.1})
        mstar_ms, mstar_q = X[-2:]
        assert np.any(mstar_ms != mstar_ms_fid)
        assert np.any(mstar_q != mstar_q_fid)


def test_in_situ_galaxy_halo_history_self_consistent_mah_dmhdt():
    """This unit-test enforces the relationship between
    Mhalo(t) and dMhalo/dt(t). We need to rescale by 1e9 since
    tarr is in Gyr but dMhalo/dt is in Msun/yr
    """
    for logM in np.linspace(10, 15, 15):
        X = in_situ_galaxy_halo_history(logM)
        tarr, mah, dmhdt = X[1:4]
        _dmdt = np.diff(mah) / np.diff(tarr)
        dmhdt_Gyr = np.insert(_dmdt, 0, _dmdt[0])
        assert np.allclose(dmhdt_Gyr / 1e9, dmhdt)


def test_in_situ_galaxy_halo_history_varies_with_qtime_params():
    """Present-day Mstar should change when each SFR param is varied."""
    params_to_vary = {
        key: value
        for key, value in DEFAULT_QTIME_PARAMS.items()
        if "scatter" not in key
    }
    mstar_ms_fid, mstar_q_fid = in_situ_galaxy_halo_history(12)[-2:]
    for key, value in params_to_vary.items():
        X = in_situ_galaxy_halo_history(12, **{key: value * 0.9 - 0.1})
        mstar_ms, mstar_q = X[-2:]
        assert np.all(mstar_ms == mstar_ms_fid)
        assert np.any(mstar_q != mstar_q_fid)


def test_in_situ_galaxy_halo_history_correctly_infers_mah_percentile_from_logtc():
    """
    """
    logm0 = 12
    logtc_med, logtk_med, dlogm_height_med = _median_mah_sigmoid_params(logm0)
    X1 = in_situ_galaxy_halo_history(logm0, logtc=logtc_med)
    X2 = in_situ_galaxy_halo_history(logm0, mah_percentile=0.5)
    mstar_ms1, mstar_ms2 = X1[-2], X2[-2]
    mstar_q1, mstar_q2 = X1[-1], X2[-1]
    assert np.allclose(mstar_ms1, mstar_ms2)
    assert np.allclose(mstar_q1, mstar_q2)


def test_in_situ_galaxy_halo_history_logtc_scatter_behavior():
    """Stellar mass should be sensitive to logtc scatter parameters
    """
    logm0 = 12
    logtc_med, __, __ = _median_mah_sigmoid_params(logm0)

    X1 = in_situ_galaxy_halo_history(logm0, logtc=logtc_med + 1,)
    X2 = in_situ_galaxy_halo_history(logm0, logtc=logtc_med - 1,)
    mstar_ms1, mstar_q1 = X1[-2:]
    mstar_ms2, mstar_q2 = X2[-2:]
    assert not np.allclose(mstar_ms1 / mstar_q1, mstar_ms2 / mstar_q2, rtol=0.01)


def test2_in_situ_galaxy_halo_history_logtc_scatter_behavior():
    """Stellar mass should not be sensitive to logtc scatter parameters
    for median growth histories
    """
    logm0 = 12
    logtc_med, __, __ = _median_mah_sigmoid_params(logm0)

    params = dict(logtc_scatter_dwarfs=0.1)
    params2 = dict(logtc_scatter_dwarfs=0.3)

    X1 = in_situ_galaxy_halo_history(logm0, mah_percentile=0.5, **params)
    X2 = in_situ_galaxy_halo_history(logm0, mah_percentile=0.5, **params2)
    mstar_ms1, mstar_q1 = X1[-2:]
    mstar_ms2, mstar_q2 = X2[-2:]
    assert np.allclose(mstar_ms1, mstar_ms2)
    assert np.allclose(mstar_q1, mstar_q2)


def test3_in_situ_galaxy_halo_history_logtc_scatter_behavior():
    """Stellar mass should be sensitive to logtc scatter parameters
    for below-average growth histories
    """
    logm0 = 12
    logtc_med, __, __ = _median_mah_sigmoid_params(logm0)

    params = dict(logtc_scatter_dwarfs=0.1)
    params2 = dict(logtc_scatter_dwarfs=0.3)

    for p in (0.1, 0.25, 0.75, 0.9):
        X1 = in_situ_galaxy_halo_history(logm0, mah_percentile=p, **params)
        X2 = in_situ_galaxy_halo_history(logm0, mah_percentile=p, **params2)
        mstar_ms1, mstar_q1 = X1[-2:]
        mstar_ms2, mstar_q2 = X2[-2:]
        assert not np.allclose(mstar_ms1, mstar_ms2)
        assert not np.allclose(mstar_q1, mstar_q2)


def test_in_situ_mstar_at_zobs_qtime_percentile_behavior():
    logm0 = 12
    X1 = in_situ_galaxy_halo_history(logm0, qtime_percentile=0)
    X2 = in_situ_galaxy_halo_history(logm0, qtime_percentile=0.5)
    X3 = in_situ_galaxy_halo_history(logm0, qtime_percentile=1)

    mstar_q1 = X1[-1]
    mstar_q2 = X2[-1]
    mstar_q3 = X3[-1]

    assert np.all(mstar_q1 <= mstar_q2)
    assert np.any(mstar_q1 < mstar_q2)
    assert np.all(mstar_q2 <= mstar_q3)
    assert np.any(mstar_q2 < mstar_q3)
