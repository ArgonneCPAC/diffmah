"""
"""
import os
import numpy as np
from ..halo_population_assembly import _get_bimodal_halo_history
from ..halo_population_assembly import LGE_ARR, LGL_ARR, X0_ARR
from ..tng_pdf_model import DEFAULT_MAH_PDF_PARAMS as TNG_PDF_PARAMS

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(os.path.dirname(_THIS_DRNAME), "data")


def test_get_average_halo_histories():
    """Verify that the _get_average_halo_histories returns reasonable arrays."""
    n_early, n_late, n_x0 = 10, 15, 20
    lge_min, lge_max = -1.5, 1.5
    lgl_min, lgl_max = -2, 1
    x0_min, x0_max = -1.0, 1
    lge_arr = np.linspace(lge_min, lge_max, n_early)
    lgl_arr = np.linspace(lgl_min, lgl_max, n_late)
    x0_arr = np.linspace(x0_min, x0_max, n_x0)
    tarr = np.linspace(1, 13.8, 25)
    lgt_arr = np.log10(tarr)
    lgmp_arr = np.array((11.25, 11.75, 12, 12.5, 13, 13.5, 14, 14.5))
    _res = _get_bimodal_halo_history(lgt_arr, lgmp_arr, lge_arr, lgl_arr, x0_arr)
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
            "14.75",
        )
    )
    lgmp_targets = np.array([float(lgm) for lgm in mlist])
    lgt = np.log10(np.load(os.path.join(DDRN, "nbody_t_target.npy")))
    mah_pat = "mean_log_mah_nbody_logmp_{}.npy"
    lgmah_fnames = list((os.path.join(DDRN, mah_pat.format(lgm)) for lgm in mlist))
    mean_log_mah_targets = np.array([np.load(fn) for fn in lgmah_fnames])

    vmah_pat = "var_log_mah_nbody_logmp_{}.npy"
    vlgmah_fnames = list((os.path.join(DDRN, vmah_pat.format(lgm)) for lgm in mlist))
    var_log_mah_targets = np.array([np.load(fn) for fn in vlgmah_fnames])

    dmhdt_pat = "mean_dmhdt_nbody_logmp_{}.npy"
    dmhdt_fnames = list((os.path.join(DDRN, dmhdt_pat.format(lgm)) for lgm in mlist))
    mean_dmhdt_targets = np.array([np.load(fn) for fn in dmhdt_fnames])

    vdmhdt_pat = "var_dmhdt_nbody_logmp_{}.npy"
    vdmhdt_fnames = list((os.path.join(DDRN, vdmhdt_pat.format(lgm)) for lgm in mlist))
    var_dmhdt_targets = np.array([np.load(fn) for fn in vdmhdt_fnames])

    _res = _get_bimodal_halo_history(lgt, lgmp_targets, LGE_ARR, LGL_ARR, X0_ARR)
    mean_dmhdt_preds, mean_log_mah_preds = _res[0], _res[2]
    var_dmhdt_preds, var_log_mah_preds = _res[3], _res[4]

    for im, lgmp in enumerate(lgmp_targets):
        x, y = mean_log_mah_targets[im, :], mean_log_mah_preds[im, :]
        msg = "Inaccurate prediction for <log10(MAH)> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(mean_dmhdt_targets[im, :]), np.log10(mean_dmhdt_preds[im, :])
        msg = "Inaccurate prediction for <dMh/dt> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.1), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = var_log_mah_targets[im, :], var_log_mah_preds[im, :]
        msg = "Inaccurate prediction for std(log10(MAH)) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.15), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(var_dmhdt_targets[im, :]), np.log10(var_dmhdt_preds[im, :])
        msg = "Inaccurate prediction for std(dMh/dt) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.15), msg.format(lgmp)


def test_average_halo_histories_agree_with_tng_simulation():
    mlist = list(
        (
            "11.75",
            "12.00",
            "12.25",
            "12.50",
            "12.75",
            "13.00",
            "13.25",
            "13.50",
        )
    )
    lgmp_targets = np.array([float(lgm) for lgm in mlist])
    lgt = np.log10(np.load(os.path.join(DDRN, "tng_t_target.npy")))
    mah_pat = "mean_log_mah_tng_logmp_{}.npy"
    lgmah_fnames = list((os.path.join(DDRN, mah_pat.format(lgm)) for lgm in mlist))
    mean_log_mah_targets = np.array([np.load(fn) for fn in lgmah_fnames])

    vmah_pat = "var_log_mah_tng_logmp_{}.npy"
    vlgmah_fnames = list((os.path.join(DDRN, vmah_pat.format(lgm)) for lgm in mlist))
    var_log_mah_targets = np.array([np.load(fn) for fn in vlgmah_fnames])

    dmhdt_pat = "mean_dmhdt_tng_logmp_{}.npy"
    dmhdt_fnames = list((os.path.join(DDRN, dmhdt_pat.format(lgm)) for lgm in mlist))
    mean_dmhdt_targets = np.array([np.load(fn) for fn in dmhdt_fnames])

    vdmhdt_pat = "var_dmhdt_tng_logmp_{}.npy"
    vdmhdt_fnames = list((os.path.join(DDRN, vdmhdt_pat.format(lgm)) for lgm in mlist))
    var_dmhdt_targets = np.array([np.load(fn) for fn in vdmhdt_fnames])

    _res = _get_bimodal_halo_history(
        lgt, lgmp_targets, LGE_ARR, LGL_ARR, X0_ARR, **TNG_PDF_PARAMS
    )
    mean_dmhdt_preds, mean_log_mah_preds = _res[0], _res[2]
    var_dmhdt_preds, var_log_mah_preds = _res[3], _res[4]

    for im, lgmp in enumerate(lgmp_targets):
        x, y = mean_log_mah_targets[im, :], mean_log_mah_preds[im, :]
        msg = "Inaccurate prediction for <log10(MAH)> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.2), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(mean_dmhdt_targets[im, :]), np.log10(mean_dmhdt_preds[im, :])
        msg = "Inaccurate prediction for <dMh/dt> at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.2), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = var_log_mah_targets[im, :], var_log_mah_preds[im, :]
        msg = "Inaccurate prediction for std(log10(MAH)) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.25), msg.format(lgmp)

    for im, lgmp in enumerate(lgmp_targets):
        x, y = np.log10(var_dmhdt_targets[im, :]), np.log10(var_dmhdt_preds[im, :])
        msg = "Inaccurate prediction for std(dMh/dt) at lgmp = {0:.2f}"
        assert np.allclose(x, y, atol=0.3), msg.format(lgmp)
