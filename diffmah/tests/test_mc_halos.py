"""
"""
import os

import numpy as np

from ..monte_carlo_halo_population import mc_halo_population
from ..tng_pdf_model import DEFAULT_MAH_PDF_PARAMS as TNG_PDF_PARAMS

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_mc_halo_assembly_namedtuple_syntax():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0 + np.zeros(n_halos)
    mc_halopop = mc_halo_population(tarr, t0, logmh)
    assert mc_halopop.dmhdt.shape == (n_halos, n_times)
    assert mc_halopop.log_mah.shape == (n_halos, n_times)
    assert mc_halopop.early_index.shape == (n_halos,)
    assert mc_halopop.late_index.shape == (n_halos,)
    assert mc_halopop.lgtc.shape == (n_halos,)
    assert mc_halopop.mah_type.shape == (n_halos,)


def test_mc_halo_assembly_returns_correctly_shaped_arrays():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0 + np.zeros(n_halos)
    res = mc_halo_population(tarr, t0, logmh)
    dmhdt, log_mah, early, late, lgtc, mah_type = res
    assert dmhdt.shape == (n_halos, n_times)
    assert log_mah.shape == (n_halos, n_times)


def test_mc_halo_assembly_returns_correct_z0_masses():
    n_halos, n_times = 500, 100
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    logmh = 12.0 + np.zeros(n_halos)
    res = mc_halo_population(tarr, t0, logmh)
    dmhdt, log_mah, early, late, lgtc, mah_type = res
    assert np.allclose(log_mah[:, -1], logmh)


def test_mc_halo_assembly_returns_monotonic_halo_histories():
    n_halos, n_times = 5000, 40
    tarr = np.linspace(0.1, 13.8, n_times)
    t0 = tarr[-1]
    for logmh in (10, 12, 14, 16):
        res = mc_halo_population(tarr, t0, logmh + np.zeros(n_halos))
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
        res = mc_halo_population(tarr, t0, logmh + np.zeros(n_halos))
        dmhdt, log_mah, early, late, lgtc, mah_type = res
        assert np.any(mah_type == 0)
        assert np.any(mah_type == 1)


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
        pop = mc_halo_population(t, t[-1], lgmp + np.zeros(n_halos))
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
        pop = mc_halo_population(t, t[-1], lgmp + np.zeros(n_halos), **TNG_PDF_PARAMS)
        dmhdt, log_mah = pop[:2]
        mean_log_mah_mc = np.mean(log_mah, axis=0)
        mean_dmhdt_mc = np.mean(dmhdt, axis=0)
        assert np.allclose(mean_log_mah_correct, mean_log_mah_mc, atol=0.15)
        assert np.allclose(mean_dmhdt_correct, mean_dmhdt_mc, rtol=0.30)
