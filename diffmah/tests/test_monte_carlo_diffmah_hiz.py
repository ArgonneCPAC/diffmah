"""
"""
import warnings

import numpy as np
from jax import random as jran

from ..individual_halo_assembly import calc_halo_history
from ..monte_carlo_diffmah_hiz import mc_diffmah_params_hiz


def test_mc_diffmah_params():
    """Enforce that mc_diffmah_params_hiz returns MAHs that pass through the input
    halo masses at the input times
    """
    t_obs = 5.5
    t0 = 13.8
    lgt0 = np.log10(t0)
    logmh = np.linspace(8, 15, 1000)
    ran_key = jran.PRNGKey(0)
    with warnings.catch_warnings(record=True) as w:
        res = mc_diffmah_params_hiz(ran_key, t_obs, logmh, lgt0=lgt0)
        assert len(w) == 0, "mc_diffmah_params_hiz raises warning"
    lgm0, lgtc, early, late = res
    tc = 10**lgtc
    tarr = np.array((t_obs, t0))
    log_mah = calc_halo_history(tarr, t0, lgm0, tc, early, late)[1]
    assert np.allclose(log_mah[:, 0], logmh, atol=1e-4)


def test_mc_diffmah_params_all_halos_with_same_mass():
    """Enforce that mc_diffmah_params_hiz returns sensible results when all input halos
    have the same mass
    """
    t_obs = 5.5
    t0 = 13.8
    lgt0 = np.log10(t0)
    logmh = np.zeros(1000) + 12
    ran_key = jran.PRNGKey(0)
    with warnings.catch_warnings(record=True) as w:
        res = mc_diffmah_params_hiz(ran_key, t_obs, logmh, lgt0=lgt0)
        assert len(w) == 0, "mc_diffmah_params_hiz raises warning"
    lgm0, lgtc, early, late = res
    tc = 10**lgtc
    tarr = np.array((t_obs, t0))
    log_mah = calc_halo_history(tarr, t0, lgm0, tc, early, late)[1]
    assert np.allclose(log_mah[:, 0], logmh, atol=1e-4)


def test_mc_diffmah_params_empty_slice_regression_test():
    """Enforce that we do not compute the median of an empty slice for a halo sample
    with high-z masses that are more coarsely spaced than our interpolation table.
    This is a regression test for a bug fixed with Pull Request
    https://github.com/ArgonneCPAC/diffmah/pull/99
    """
    t_obs = 13.4
    t0 = 13.8
    lgt0 = np.log10(t0)
    logmh = np.array((10.84, 11.2))
    ran_key = jran.PRNGKey(0)
    with warnings.catch_warnings(record=True) as w:
        res = mc_diffmah_params_hiz(ran_key, t_obs, logmh, lgt0=lgt0)
        assert len(w) == 0, "mc_diffmah_params_hiz raises warning"
    lgm0, lgtc, early, late = res
    tc = 10**lgtc
    tarr = np.array((t_obs, t0))
    log_mah = calc_halo_history(tarr, t0, lgm0, tc, early, late)[1]
    assert np.allclose(log_mah[:, 0], logmh, atol=1e-4)
