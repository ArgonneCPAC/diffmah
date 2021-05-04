"""
"""
import os

import numpy as np
from ..monte_carlo_halo_population import mc_halo_population
from ..tng_pdf_model import DEFAULT_MAH_PDF_PARAMS as TNG_PDF_PARAMS

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_mc_halo_assembly_returns_correctly_shaped_arrays():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0
    res = mc_halo_population(tarr, t0, logmh, n_halos)
    dmhdt, log_mah, early, late, lgtc, mah_type = res
    assert dmhdt.shape == (n_halos, n_times)
    assert log_mah.shape == (n_halos, n_times)


def test_mc_halo_assembly_returns_correct_z0_masses():
    n_halos, n_times = 500, 100
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0
    res = mc_halo_population(tarr, t0, logmh, n_halos)
    dmhdt, log_mah, early, late, lgtc, mah_type = res
    assert np.allclose(log_mah[:, -1], logmh)


def test_mc_halo_assembly_returns_monotonic_halo_histories():
    n_halos, n_times = 5000, 40
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    for logmh in (10, 12, 14, 16):
        res = mc_halo_population(tarr, t0, logmh, n_halos)
        dmhdt, log_mah, early, late, lgtc, mah_type = res

        msg = "Some MC generated halos have non-increasing masses across time"
        delta_log_mah = np.diff(log_mah, axis=1)
        neg_msk = np.any(delta_log_mah < 0, axis=1)
        assert neg_msk.sum() == 0, msg
        assert dmhdt.min() >= 0, msg

        msg2 = "Some MC generated halos have early_index <= late_index"
        assert np.all(early > late), msg2


def test_mc_halo_assembly_returns_mix_of_halo_types():
    n_halos, n_times = 500, 10
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    for logmh in (10, 12, 14, 16):
        res = mc_halo_population(tarr, t0, logmh, n_halos)
        dmhdt, log_mah, early, late, lgtc, mah_type = res
        assert np.any(mah_type == "early")
        assert np.any(mah_type == "late")


def test_mah_type_argument_of_mc_halo_assembly():
    n_halos, n_times = 50, 10
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    for logmh in (10, 12, 14, 16):
        res0 = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="early")
        dmhdt0, log_mah0, early0, late0, lgtc0, mah_type0 = res0
        assert np.all(mah_type0 == "early")
        res1 = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="late")
        dmhdt1, log_mah1, early1, late1, lgtc1, mah_type1 = res1
        assert np.all(mah_type1 == "late")
        assert lgtc0.mean() < lgtc1.mean()


def test_mc_halo_assembly_early_late_options():
    n_halos, n_times = int(5e4), 20
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]

    for logmh in (11, 13, 15):
        res = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="early")
        dmhdt_early, log_mah_early, early_early, late_early, lgtc_early, mah_type = res
        assert dmhdt_early.shape == (n_halos, n_times)
        assert log_mah_early.shape == (n_halos, n_times)
        assert np.allclose(log_mah_early[:, -1], logmh)
        assert np.all(mah_type == "early")

        res = mc_halo_population(tarr, t0, logmh, n_halos, mah_type="late")
        dmhdt_late, log_mah_late, early_late, late_late, lgtc_late, mah_type = res
        assert dmhdt_late.shape == (n_halos, n_times)
        assert log_mah_late.shape == (n_halos, n_times)
        assert np.allclose(log_mah_late[:, -1], logmh)
        assert np.all(mah_type == "late")

        assert np.mean(lgtc_early) < np.mean(lgtc_late)

        res = mc_halo_population(tarr, t0, logmh, n_halos)
        dmhdt, log_mah, early, late, lgtc, mah_type = res

        mean_dmhdt_early = np.mean(dmhdt_early, axis=0)
        mean_dmhdt_late = np.mean(dmhdt_late, axis=0)
        mean_dmhdt_early2 = np.mean(dmhdt[mah_type == "early"], axis=0)
        mean_dmhdt_late2 = np.mean(dmhdt[mah_type == "late"], axis=0)
        assert np.allclose(mean_dmhdt_early, mean_dmhdt_early2, rtol=0.1)
        assert np.allclose(mean_dmhdt_late, mean_dmhdt_late2, rtol=0.1)

        mean_log_mah_early = np.mean(log_mah_early, axis=0)
        mean_log_mah_late = np.mean(log_mah_late, axis=0)
        mean_log_mah_early2 = np.mean(log_mah[mah_type == "early"], axis=0)
        mean_log_mah_late2 = np.mean(log_mah[mah_type == "late"], axis=0)
        assert np.allclose(mean_log_mah_early, mean_log_mah_early2, rtol=0.1)
        assert np.allclose(mean_log_mah_late, mean_log_mah_late2, rtol=0.1)


def test_mc_realization_agrees_with_tabulated_nbody_simulation_data():
    mlist = ("12.00", "13.00", "14.00")
    lgmp_targets = np.array([float(lgm) for lgm in mlist])
    t = np.loadtxt(os.path.join(DDRN, "nbody_t_target.dat"))
    mah_pat = "mean_log_mah_nbody_logmp_{}.dat"
    lgmah_fnames = list((os.path.join(DDRN, mah_pat.format(lgm)) for lgm in mlist))
    mean_log_mah_targets = np.array([np.loadtxt(fn) for fn in lgmah_fnames])

    dmhdt_pat = "mean_dmhdt_nbody_logmp_{}.dat"
    dmhdt_fnames = list((os.path.join(DDRN, dmhdt_pat.format(lgm)) for lgm in mlist))
    mean_dmhdt_targets = np.array([np.loadtxt(fn) for fn in dmhdt_fnames])

    n_halos = int(1e4)
    for im, lgmp in enumerate(lgmp_targets):
        mean_log_mah_correct = mean_log_mah_targets[im]
        mean_dmhdt_correct = mean_dmhdt_targets[im]
        pop = mc_halo_population(t, t[-1], lgmp, n_halos)
        dmhdt, log_mah = pop[:2]
        mean_log_mah_mc = np.mean(log_mah, axis=0)
        mean_dmhdt_mc = np.mean(dmhdt, axis=0)
        assert np.allclose(mean_log_mah_correct, mean_log_mah_mc, atol=0.15)
        assert np.allclose(mean_dmhdt_correct, mean_dmhdt_mc, rtol=0.30)


def test_mc_realization_agrees_with_tabulated_tng_simulation_data():
    mlist = ("12.00", "13.00")
    lgmp_targets = np.array([float(lgm) for lgm in mlist])
    t = np.loadtxt(os.path.join(DDRN, "tng_t_target.dat"))
    mah_pat = "mean_log_mah_tng_logmp_{}.dat"
    lgmah_fnames = list((os.path.join(DDRN, mah_pat.format(lgm)) for lgm in mlist))
    mean_log_mah_targets = np.array([np.loadtxt(fn) for fn in lgmah_fnames])

    dmhdt_pat = "mean_dmhdt_tng_logmp_{}.dat"
    dmhdt_fnames = list((os.path.join(DDRN, dmhdt_pat.format(lgm)) for lgm in mlist))
    mean_dmhdt_targets = np.array([np.loadtxt(fn) for fn in dmhdt_fnames])

    n_halos = int(1e4)
    for im, lgmp in enumerate(lgmp_targets):
        mean_log_mah_correct = mean_log_mah_targets[im]
        mean_dmhdt_correct = mean_dmhdt_targets[im]
        pop = mc_halo_population(t, t[-1], lgmp, n_halos, **TNG_PDF_PARAMS)
        dmhdt, log_mah = pop[:2]
        mean_log_mah_mc = np.mean(log_mah, axis=0)
        mean_dmhdt_mc = np.mean(dmhdt, axis=0)
        assert np.allclose(mean_log_mah_correct, mean_log_mah_mc, atol=0.15)
        assert np.allclose(mean_dmhdt_correct, mean_dmhdt_mc, rtol=0.30)
