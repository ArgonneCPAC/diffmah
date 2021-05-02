"""
"""
import os
import numpy as np
from ..halo_population_assembly import _get_bimodal_halo_history
from ..halo_population_assembly import UE_ARR, UL_ARR, LGTC_ARR
from ..tng_pdf_model import DEFAULT_MAH_PDF_PARAMS as TNG_PARAMS

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_get_average_halo_histories():
    """Verify that the _get_average_halo_histories returns reasonable arrays."""
    tarr = np.linspace(1, 13.8, 25)
    lgt_arr = np.log10(tarr)
    lgmp_arr = np.array((11.25, 11.75, 12, 12.5, 13, 13.5, 14, 14.5))
    _res = _get_bimodal_halo_history(lgt_arr, lgmp_arr, UE_ARR, UL_ARR, LGTC_ARR)
    mean_dmhdt, mean_mah, mean_log_mah, variance_dmhdt, variance_mah = _res
    mean_log_mahs = np.log10(mean_mah)

    #  Average halo MAHs should agree at t=today
    assert np.allclose(mean_log_mahs[:, -1], lgmp_arr, atol=0.01)

    # Average halo MAHs should monotonically increase
    assert np.all(np.diff(mean_log_mahs, axis=1) > 0)

    # Average halo accretion rates should monotonically increase with present-day mass
    assert np.all(np.diff(mean_dmhdt[:, -1]) > 0)


def test_average_halo_histories_agree_with_nbody_simulations():
    mlist = list(
        (
            "11.50",
            "11.75",
            "12.00",
            "12.25",
            "12.50",
            "12.75",
            "13.00",
            "13.25",
            "13.50",
            "13.75",
            "14.00",
            "14.25",
            "14.50",
        )
    )
    lgmp_targets = np.array([float(lgm) for lgm in mlist])
    lgt = np.log10(np.loadtxt(os.path.join(DDRN, "nbody_t_target.dat")))
    mah_pat = "mean_log_mah_nbody_logmp_{}.dat"
    lgmah_fnames = list((os.path.join(DDRN, mah_pat.format(lgm)) for lgm in mlist))
    mean_log_mah_targets = np.array([np.loadtxt(fn) for fn in lgmah_fnames])

    vmah_pat = "var_log_mah_nbody_logmp_{}.dat"
    vlgmah_fnames = list((os.path.join(DDRN, vmah_pat.format(lgm)) for lgm in mlist))
    var_log_mah_targets = np.array([np.loadtxt(fn) for fn in vlgmah_fnames])

    dmhdt_pat = "mean_dmhdt_nbody_logmp_{}.dat"
    dmhdt_fnames = list((os.path.join(DDRN, dmhdt_pat.format(lgm)) for lgm in mlist))
    mean_dmhdt_targets = np.array([np.loadtxt(fn) for fn in dmhdt_fnames])

    vdmhdt_pat = "var_dmhdt_nbody_logmp_{}.dat"
    vdmhdt_fnames = list((os.path.join(DDRN, vdmhdt_pat.format(lgm)) for lgm in mlist))
    var_dmhdt_targets = np.array([np.loadtxt(fn) for fn in vdmhdt_fnames])

    _res = _get_bimodal_halo_history(lgt, lgmp_targets, UE_ARR, UL_ARR, LGTC_ARR)
    mean_dmhdt_preds, mean_log_mah_preds = _res[0], _res[2]
    var_dmhdt_preds, var_log_mah_preds = _res[3], _res[4]

    for im, lgmp in enumerate(lgmp_targets):
        x, y = mean_log_mah_targets[im, :], mean_log_mah_preds[im, :]
        msg = "Inaccurate N-body prediction for <log10(MAH)> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(mean_dmhdt_targets[im, :]), np.log10(mean_dmhdt_preds[im, :])
        msg = "Inaccurate N-body prediction for <dMh/dt> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = var_log_mah_targets[im, :], var_log_mah_preds[im, :]
        msg = "Inaccurate N-body prediction for std(log10(MAH)) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(var_dmhdt_targets[im, :]), np.log10(var_dmhdt_preds[im, :])
        msg = "Inaccurate N-body prediction for std(dMh/dt) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)


def test_average_halo_histories_agree_with_tng():
    mlist = list(
        (
            "11.50",
            "11.75",
            "12.00",
            "12.25",
            "12.50",
            "12.75",
            "13.00",
            "13.25",
            "13.50",
            "13.75",
        )
    )
    lgmp_targets = np.array([float(lgm) for lgm in mlist])
    lgt = np.log10(np.loadtxt(os.path.join(DDRN, "tng_t_target.dat")))
    mah_pat = "mean_log_mah_tng_logmp_{}.dat"
    lgmah_fnames = list((os.path.join(DDRN, mah_pat.format(lgm)) for lgm in mlist))
    mean_log_mah_targets = np.array([np.loadtxt(fn) for fn in lgmah_fnames])

    vmah_pat = "var_log_mah_tng_logmp_{}.dat"
    vlgmah_fnames = list((os.path.join(DDRN, vmah_pat.format(lgm)) for lgm in mlist))
    var_log_mah_targets = np.array([np.loadtxt(fn) for fn in vlgmah_fnames])

    dmhdt_pat = "mean_dmhdt_tng_logmp_{}.dat"
    dmhdt_fnames = list((os.path.join(DDRN, dmhdt_pat.format(lgm)) for lgm in mlist))
    mean_dmhdt_targets = np.array([np.loadtxt(fn) for fn in dmhdt_fnames])

    vdmhdt_pat = "var_dmhdt_tng_logmp_{}.dat"
    vdmhdt_fnames = list((os.path.join(DDRN, vdmhdt_pat.format(lgm)) for lgm in mlist))
    var_dmhdt_targets = np.array([np.loadtxt(fn) for fn in vdmhdt_fnames])

    _res = _get_bimodal_halo_history(
        lgt, lgmp_targets, UE_ARR, UL_ARR, LGTC_ARR, **TNG_PARAMS
    )
    mean_dmhdt_preds, mean_log_mah_preds = _res[0], _res[2]
    var_dmhdt_preds, var_log_mah_preds = _res[3], _res[4]

    for im, lgmp in enumerate(lgmp_targets):
        x, y = mean_log_mah_targets[im, :], mean_log_mah_preds[im, :]
        msg = "Inaccurate TNG prediction for <log10(MAH)> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(mean_dmhdt_targets[im, :]), np.log10(mean_dmhdt_preds[im, :])
        msg = "Inaccurate TNG prediction for <dMh/dt> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = var_log_mah_targets[im, :], var_log_mah_preds[im, :]
        msg = "Inaccurate TNG prediction for std(log10(MAH)) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(var_dmhdt_targets[im, :]), np.log10(var_dmhdt_preds[im, :])
        msg = "Inaccurate TNG prediction for std(dMh/dt) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.2), msg.format(lgmp)
