"""
"""
import pytest
import numpy as np
from astropy.cosmology import Planck15
from collections import OrderedDict
from ..in_situ_smh_kernel import _get_model_param_dictionaries
from ..in_situ_smh_kernel import in_situ_mstar_at_zobs
from ..sigmoid_mah import DEFAULT_MAH_PARAMS


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


def test1_in_situ_stellar_mass_at_zobs_is_monotonic_in_mass():
    for z in (0, 1, 2, 5):
        mstar_ms10, mstar_q10 = in_situ_mstar_at_zobs(z, 10)
        mstar_ms12, mstar_q12 = in_situ_mstar_at_zobs(z, 12)
        mstar_ms14, mstar_q14 = in_situ_mstar_at_zobs(z, 14)
        assert mstar_ms10 < mstar_ms12 < mstar_ms14
        assert mstar_q10 < mstar_q12 < mstar_q14


def test2_in_situ_stellar_mass_at_zobs_is_monotonic_in_mass():
    logmarr = np.linspace(8, 17, 20)

    for z in (0, 1, 2, 5):
        mstar_ms_last, __ = in_situ_mstar_at_zobs(z, logmarr[0])
        for logm in logmarr[1:]:
            mstar_ms, mstar_q = in_situ_mstar_at_zobs(z, logm)
            assert mstar_ms >= mstar_q
            assert mstar_ms > mstar_ms_last
            mstar_ms_last = mstar_ms


def test_in_situ_stellar_mass_at_zobs_scales_correctly_with_mah_percentile():
    """Earlier-forming halos have greater M* today."""
    zobs = 0
    mstar_ms_1, __ = in_situ_mstar_at_zobs(zobs, 12, mah_percentile=0)
    mstar_ms_2, __ = in_situ_mstar_at_zobs(zobs, 12)
    mstar_ms_3, __ = in_situ_mstar_at_zobs(zobs, 12, mah_percentile=1)
    assert mstar_ms_1 > mstar_ms_2 > mstar_ms_3


def test_in_situ_stellar_mass_at_zobs_scales_correctly_with_logtc():
    """Earlier-forming halos have greater M* today."""
    zobs = 0
    mstar_ms_1, __ = in_situ_mstar_at_zobs(zobs, 12, logtc=-0.5)
    mstar_ms_2, __ = in_situ_mstar_at_zobs(zobs, 12, logtc=0)
    mstar_ms_3, __ = in_situ_mstar_at_zobs(zobs, 12, logtc=0.5)
    assert mstar_ms_1 > mstar_ms_2 > mstar_ms_3


def test_in_situ_mstar_at_zobs_catches_bad_mah_percentile_inputs():
    zobs, logm0 = 0, 12
    with pytest.raises(ValueError):
        in_situ_mstar_at_zobs(zobs, logm0, mah_percentile=1, logtc=1)


def test_in_situ_mstar_at_zobs_sensible_quenching_behavior():
    logmarr = np.linspace(8, 17, 20)
    for z in (0, 1, 2, 5):
        for logm in logmarr:
            mstar_ms, mstar_q = in_situ_mstar_at_zobs(z, logm)
            assert mstar_ms >= mstar_q


def test_in_situ_mstar_at_zobs_sensible_qtime_behavior():
    """When qtime > today, quenching should not change present-day M*."""
    zobs, logm0 = 0, 12
    tobs = 13.8
    mstar_ms, mstar_q = in_situ_mstar_at_zobs(zobs, logm0, qtime=tobs + 1)
    assert mstar_q > mstar_ms * 0.9


def test2_in_situ_mstar_at_zobs_sensible_qtime_behavior():
    """When qtime < today, quenching should not change present-day M*."""
    zobs, logm0 = 0, 12
    tobs = 13.8
    mstar_ms, mstar_q = in_situ_mstar_at_zobs(zobs, logm0, qtime=tobs - 1)
    assert mstar_q < mstar_ms * 0.9


def tes3_in_situ_mstar_at_zobs_sensible_qtime_behavior():
    """When qtime > tobs, quenching should not change M*(tobs)."""
    zobs, logm0 = 1, 12
    tobs = Planck15.age(zobs).value  # roughly 5.9 Gyr
    mstar_ms, mstar_q = in_situ_mstar_at_zobs(zobs, logm0, qtime=tobs + 1)
    assert mstar_q > mstar_ms * 0.9


def test4_in_situ_mstar_at_zobs_sensible_qtime_behavior():
    """When qtime < tobs, quenching should significantly reduce M*(tobs)."""
    zobs, logm0 = 1, 12
    tobs = Planck15.age(zobs).value  # roughly 5.9 Gyr
    mstar_ms, mstar_q = in_situ_mstar_at_zobs(zobs, logm0, qtime=tobs - 1)
    assert mstar_q < mstar_ms * 0.9


def test_in_situ_mstar_at_zobs_varies_with_MAH_params():
    """Present-day Mstar should change when each MAH param is varied."""
    zobs, logm0 = 0, 12
    mstar_ms_fid, mstar_q_fid = in_situ_mstar_at_zobs(zobs, logm0)
    mah_params_to_vary = {
        key: value for key, value in DEFAULT_MAH_PARAMS.items() if "scatter" not in key
    }
    for key, value in mah_params_to_vary.items():
        mstar_ms_alt, mstar_q_alt = in_situ_mstar_at_zobs(
            zobs, logm0, **{key: value * 0.9 - 0.1}
        )
        pat = "parameter '{0}' has no effect on {1}"
        assert mstar_ms_alt != mstar_ms_fid, pat.format(key, "mstar_ms")
        assert mstar_q_alt != mstar_q_fid, pat.format(key, "mstar_q")
