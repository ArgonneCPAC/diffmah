"""
"""
import pytest
import numpy as np
from collections import OrderedDict
from ..in_situ_smh_kernel import _get_model_param_dictionaries
from ..in_situ_smh_kernel import in_situ_mstar_at_zobs


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
        result = _get_model_param_dictionaries(*defaults, **new_vals)


def test4_get_model_param_dictionaries():
    defaults = [OrderedDict(a=1, b=2), OrderedDict(a=3, d=4)]
    new_vals = OrderedDict(d=5)
    with pytest.raises(KeyError):
        result = _get_model_param_dictionaries(*defaults, **new_vals)


def test1_in_situ_stellar_mass_at_zobs_is_monotonic_in_mass():
    for z in (0, 1, 2, 5):
        mstar_ms10, mstar_med10, mstar_q10 = in_situ_mstar_at_zobs(z, 10)
        mstar_ms12, mstar_med12, mstar_q12 = in_situ_mstar_at_zobs(z, 12)
        mstar_ms14, mstar_med14, mstar_q14 = in_situ_mstar_at_zobs(z, 14)
        assert mstar_ms10 < mstar_ms12 < mstar_ms14
        assert mstar_med10 < mstar_med12 < mstar_med14
        assert mstar_q10 < mstar_q12 < mstar_q14


def test2_in_situ_stellar_mass_at_zobs_is_monotonic_in_mass():
    logmarr = np.linspace(8, 17, 20)

    for z in (0, 1, 2, 5):
        mstar_ms_last, mstar_med_last, __ = in_situ_mstar_at_zobs(z, logmarr[0])
        for logm in logmarr[1:]:
            mstar_ms, mstar_med, mstar_q = in_situ_mstar_at_zobs(z, logm)
            assert mstar_ms >= mstar_med >= mstar_q
            assert mstar_ms > mstar_ms_last
            assert mstar_med > mstar_med_last
            mstar_ms_last, mstar_med_last = mstar_ms, mstar_med
