"""
"""
import numpy as np
from ..sigmoid_mah import median_logmpeak_from_logt, logmpeak_from_logt
from ..sigmoid_mah import logm0_from_logm_at_logt
from ..sigmoid_mah import _median_mah_sigmoid_params, logtc_from_logm_at_logt
from ..sigmoid_mah import _mah_sigmoid_params_logm_at_logt
from ..sigmoid_mah import _logtc_from_mah_percentile


def test1_mah_sigmoid_params_logm_at_logt():
    logt = 0.8
    logm_at_logt = 12
    mah_params = {}
    logtc, logtk, dlogm_height, logm0 = _mah_sigmoid_params_logm_at_logt(
        logt, logm_at_logt, logtc=None, **mah_params
    )


def test2_mah_sigmoid_params_logm_at_logt():
    logt = 0.8
    logm_at_logt = np.linspace(10, 15, 20)
    mah_params = {}
    logtc, logtk, dlogm_height, logm0 = _mah_sigmoid_params_logm_at_logt(
        logt, logm_at_logt, logtc=None, **mah_params
    )


def test_median_logmpeak_from_logt_is_monotonic_at_z0():
    logt = np.log10(np.linspace(0.1, 10 ** 1.14, 1000))
    logmah5 = median_logmpeak_from_logt(logt, 5, logt0=1.14)
    logmah10 = median_logmpeak_from_logt(logt, 10, logt0=1.14)
    logmah15 = median_logmpeak_from_logt(logt, 15, logt0=1.14)
    assert logmah5[-1] < logmah10[-1] < logmah15[-1]
    assert np.all(np.diff(logmah5) > 0)
    assert np.all(np.diff(logmah10) > 0)
    assert np.all(np.diff(logmah15) > 0)


def test_median_mah_sigmoid_params_responds_to_input_params():
    logm0arr = np.linspace(10, 15, 50)
    logtc1, logtk1, dlogm_height1 = _median_mah_sigmoid_params(logm0arr)
    logtc2, logtk2, dlogm_height2 = _median_mah_sigmoid_params(logm0arr, logtc_logm0=10)
    assert not np.allclose(logtc1, logtc2)
    assert np.allclose(logtk1, logtk2)

    logtc3, logtk3, dlogm_height3 = _median_mah_sigmoid_params(logm0arr, dlogm_height=4)
    assert not np.allclose(dlogm_height1, dlogm_height3)
    assert np.allclose(logtc1, logtc3)


def test_median_logmpeak_from_logt_behaves_properly_with_logt0():
    logt = np.log10(np.linspace(0.1, 10 ** 1.14, 1000))
    logmah10 = median_logmpeak_from_logt(logt, 10, 1)
    logmah10b = median_logmpeak_from_logt(logt, 10, 1.25)
    assert logmah10b[-1] < logmah10[-1]


def test_logmpeak_from_logt_is_monotonic():
    logt = np.log10(np.linspace(0.1, 10 ** 1.14, 1000))
    logtc, k, dlogm_height = 0, 3, 5
    logmpeak_at_logt0, logt0 = 15, 1.14
    logmah = logmpeak_from_logt(logt, logtc, k, dlogm_height, logmpeak_at_logt0, logt0)
    assert np.all(np.diff(logmah) > 0)


def test_logm0_from_logm_at_logt():
    """
    """
    logt0 = 1.14
    logtc, logtk, dlogm_height = 0, 3, 5

    logtobs_arr = np.linspace(0, logt0, 20)
    logm0_arr = np.linspace(9, 16, 50)
    for logtobs in logtobs_arr:
        for logm0 in logm0_arr:
            logm_at_logt = logmpeak_from_logt(
                logtobs, logtc, logtk, dlogm_height, logm0, logt0
            )
            pred_logm0 = logm0_from_logm_at_logt(
                logtobs, logm_at_logt, logtc, logtk, dlogm_height, logt0
            )
            assert np.allclose(pred_logm0, logm0, rtol=0.02)


def test_logtc_from_logm_at_logt_correctly_inverts():
    logm0arr = np.linspace(10, 15, 50)
    logtc_correct = _median_mah_sigmoid_params(logm0arr)[0]

    logtobs = 0.75
    logm_at_logt = median_logmpeak_from_logt(logtobs, logm0arr)
    logtc_inferred = logtc_from_logm_at_logt(logtobs, logm_at_logt)
    assert np.allclose(logtc_inferred, logtc_correct, rtol=0.02)


def test_logtc_from_logm_at_logt_changes_with_params():
    logm0arr = np.linspace(10, 15, 50)

    for logtobs in (0, 0.25, 0.5, 0.75, 1):
        logm_at_logt = median_logmpeak_from_logt(logtobs, logm0arr)
        logtc = logtc_from_logm_at_logt(logtobs, logm_at_logt)
        logtc2 = logtc_from_logm_at_logt(logtobs, logm_at_logt, logtc_logm0=9)
        logtc3 = logtc_from_logm_at_logt(logtobs, logm_at_logt, dlogm_height=3)
        assert not np.allclose(logtc, logtc2)
        assert np.allclose(logtc, logtc3)


def test_mah_sigmoid_params_logm_at_logt():
    logt, logm_at_logt = 0.75, 12
    logtc, logtk, dlogm_height, logm0 = _mah_sigmoid_params_logm_at_logt(
        logt, logm_at_logt, logtc=-0.25
    )
    assert np.allclose(logtc, -0.25)


def test_logtc_from_mah_percentile_fiducial_model():
    logm0, p = 12, 0.5
    logtc_med = _logtc_from_mah_percentile(logm0, p)
    logtc_med_correct, __, __ = _median_mah_sigmoid_params(logm0)
    assert np.allclose(logtc_med, logtc_med_correct)


def test_logtc_from_mah_percentile_varies_with_params():
    logm0, p = 12, 0.25
    params = dict(logtc_scatter_dwarfs=1, logtc_scatter_clusters=1)
    logtc_med_fid = _logtc_from_mah_percentile(logm0, p)
    logtc_med_alt = _logtc_from_mah_percentile(logm0, p, **params)
    assert not np.allclose(logtc_med_fid, logtc_med_alt)


def test_logtc_from_mah_percentile_varies_with_percentile_correctly():
    logm0, p = 12, 0.25
    params = dict(logtc_scatter_dwarfs=1, logtc_scatter_clusters=1)
    logtc_lo = _logtc_from_mah_percentile(logm0, 0, **params)
    logtc_hi = _logtc_from_mah_percentile(logm0, 1, **params)
    logtc_med, __, __ = _median_mah_sigmoid_params(logm0, **params)
    assert logtc_lo == logtc_med - 1
    assert logtc_hi == logtc_med + 1


def test2_logtc_from_mah_percentile_varies_with_percentile_correctly():
    params = dict(logtc_scatter_dwarfs=0.1)
    logtc_lo = _logtc_from_mah_percentile(0, 0, **params)
    logtc_med = _logtc_from_mah_percentile(0, 0.5, **params)
    logtc_hi = _logtc_from_mah_percentile(0, 1, **params)
    assert np.allclose(logtc_lo, logtc_med - params["logtc_scatter_dwarfs"], rtol=1e-3)
    assert np.allclose(logtc_hi, logtc_med + params["logtc_scatter_dwarfs"], rtol=1e-3)
