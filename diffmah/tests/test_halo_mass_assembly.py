"""
"""
import numpy as np
from scipy.integrate import trapz
from ..halo_mass_assembly import halo_mass_vs_time, halo_dmdt_vs_time
from ..halo_mass_assembly import _get_mah_sigmoid_params, _logtc_from_mah_percentile
from ..sigmoid_mah import logmpeak_from_logt as logmpeak_from_logt_np


def test_jax_mah_model_agrees_with_numpy_sigmoid_mah():
    """Enforce new implementation agrees with old."""
    logtk, dlogm_height = 3, 5
    logt0 = 1.14
    t0 = 10 ** logt0
    logtc = 0

    for logm0 in np.arange(5, 17):
        time = np.linspace(0.1, t0, 35)
        logt = np.log10(time)

        logmpeak_np = logmpeak_from_logt_np(
            logt, logtc, logtk, dlogm_height, logm0, logt0
        )

        logmpeak_jax = halo_mass_vs_time(
            time, logm0, t0=t0, logtc=logtc, logtk=logtk, dlogm_height=dlogm_height,
        )
        assert np.allclose(logmpeak_np, logmpeak_jax, rtol=0.001)


def test_halo_dmdt_vs_time_integrates_to_halo_mass_vs_time():
    t0 = 13.8
    time = np.linspace(0.1, t0, 200)

    p0 = dict(t0=t0, logtc=0, logtk=3, dlogm_height=5)
    p1 = dict(t0=t0, logtc=0.5, logtk=3, dlogm_height=5)
    p2 = dict(t0=t0, logtc=0, logtk=6, dlogm_height=5)
    p3 = dict(t0=t0, logtc=0, logtk=3, dlogm_height=2)
    p4 = dict(mah_percentile=0.1)

    param_list = [p0, p1, p2, p3, p4]
    for params in param_list:
        for logm0 in np.linspace(8, 16.5, 25):
            logmpeak_jax = halo_mass_vs_time(time, logm0, **params,)
            dmhdt_jax = halo_dmdt_vs_time(time, logm0, **params,)
            assert logmpeak_jax.shape == dmhdt_jax.shape

            integrated_logmh = np.log10(trapz(dmhdt_jax, x=time)) + 9
            assert np.allclose(integrated_logmh, logm0, rtol=0.001)


def test_median_logmhalo_is_monotonic_at_z0():
    today = 13.8
    t = np.linspace(0.1, today, 100)
    logmah5 = halo_mass_vs_time(t, 5, t0=today)
    logmah10 = halo_mass_vs_time(t, 10, t0=today)
    logmah15 = halo_mass_vs_time(t, 15, t0=today)
    assert logmah5[-1] < logmah10[-1] < logmah15[-1]
    assert np.all(np.diff(logmah5) > 0)
    assert np.all(np.diff(logmah10) > 0)
    assert np.all(np.diff(logmah15) > 0)


def test_median_mah_sigmoid_params_responds_to_input_params():
    logm0arr = np.linspace(10, 15, 50)
    logtc1, logtk1, dlogm_height1 = _get_mah_sigmoid_params(logm0arr)
    logtc2, logtk2, dlogm_height2 = _get_mah_sigmoid_params(logm0arr, logtc_logm0=10)
    assert not np.allclose(logtc1, logtc2)
    assert np.allclose(logtk1, logtk2)

    logtc3, logtk3, dlogm_height3 = _get_mah_sigmoid_params(logm0arr, dlogm_height=4)
    assert not np.allclose(dlogm_height1, dlogm_height3)
    assert np.allclose(logtc1, logtc3)


def test_halo_mass_behaves_properly_with_t0():
    today = 13.8
    logm0 = 13.0
    t = np.linspace(0.1, today, 100)
    logmah1 = halo_mass_vs_time(t, logm0, 13)
    logmah2 = halo_mass_vs_time(t, logm0, 14)
    assert logmah2[-1] < logmah1[-1]


def test_halo_mass_vs_time_is_monotonic_in_time():
    logm0 = 12
    for logm0 in np.arange(5, 17):
        t = np.linspace(0.1, 10 ** 1.14, 30)
        mah = 10 ** halo_mass_vs_time(t, logm0)
        assert np.all(np.diff(mah) > 0)


def test_logtc_from_mah_percentile_fiducial_model():
    for logm0 in np.arange(5, 17):
        logtc_med = _logtc_from_mah_percentile(logm0, 0.5)
        logtc_med_correct, __, __ = _get_mah_sigmoid_params(logm0)
        assert np.allclose(logtc_med, logtc_med_correct, rtol=0.001)


def test_logtc_from_mah_percentile_varies_with_params():
    logm0, p = 12, 0.25
    params = dict(logtc_scatter_dwarfs=1, logtc_scatter_clusters=1)
    logtc_med_fid = _logtc_from_mah_percentile(logm0, p)
    logtc_med_alt = _logtc_from_mah_percentile(logm0, p, **params)
    assert not np.allclose(logtc_med_fid, logtc_med_alt)


def test_logtc_from_mah_percentile_varies_with_percentile_correctly():
    params = dict(logtc_scatter_dwarfs=1, logtc_scatter_clusters=1)
    for logm0 in np.arange(5, 17):
        logtc_lo = _logtc_from_mah_percentile(logm0, 0, **params)
        logtc_hi = _logtc_from_mah_percentile(logm0, 1, **params)
        logtc_med, __, __ = _get_mah_sigmoid_params(logm0, **params)
        assert np.allclose(logtc_lo, logtc_med - 1, rtol=0.001)
        assert np.allclose(logtc_hi, logtc_med + 1, rtol=0.001)


def test2_logtc_from_mah_percentile_varies_with_percentile_correctly():
    params = dict(logtc_scatter_dwarfs=0.1)
    logtc_lo = _logtc_from_mah_percentile(0, 0, **params)
    logtc_med = _logtc_from_mah_percentile(0, 0.5, **params)
    logtc_hi = _logtc_from_mah_percentile(0, 1, **params)
    assert np.allclose(logtc_lo, logtc_med - params["logtc_scatter_dwarfs"], rtol=0.01)
    assert np.allclose(logtc_hi, logtc_med + params["logtc_scatter_dwarfs"], rtol=0.01)
