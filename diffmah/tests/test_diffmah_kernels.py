""" """

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import DEFAULT_MAH_PARAMS as OLD_DEFAULT_MAH_PARAMS
from .. import diffmah_kernels as dk
from .. import mah_singlehalo as old_mah_singlehalo

TOL = 1e-3


def test_all_convenience_kernels_import_from_toplevel():
    from .. import logmh_at_t_obs, mah_halopop, mah_singlehalo  # noqa


def test_dk_kern_agrees_with_old_diffmah():
    tarr = np.linspace(0.1, 13.8, 200)
    logt0 = np.log10(tarr[-1])

    old_mah_params = OLD_DEFAULT_MAH_PARAMS
    dmhdt, log_mah = old_mah_singlehalo(old_mah_params, tarr, logt0)

    new_mah_params = dk.DEFAULT_MAH_PARAMS
    dmhdt_new, log_mah_new = dk._diffmah_kern(new_mah_params, tarr, logt0)

    assert np.allclose(dmhdt, dmhdt_new, rtol=1e-4)
    assert np.allclose(log_mah, log_mah_new, rtol=1e-4)


def test_dk_kern_t_q_behaves_as_expected():

    tarr = np.linspace(0.01, 13.8, 1_000)
    logt0 = np.log10(tarr[-1])

    old_mah_params = OLD_DEFAULT_MAH_PARAMS
    dmhdt_old, log_mah_old = old_mah_singlehalo(old_mah_params, tarr, logt0)

    # enforce t_peak has correct sign of effect
    new_mah_params = dk.DEFAULT_MAH_PARAMS._replace(t_peak=2.0)
    dmhdt_new, log_mah_new = dk._diffmah_kern(new_mah_params, tarr, logt0)

    assert not np.allclose(dmhdt_old, dmhdt_new, rtol=1e-4)
    assert not np.allclose(log_mah_old, log_mah_new, rtol=1e-4)

    epsilon = 1e-4
    assert np.all(log_mah_new - log_mah_old < epsilon)

    # enforce smaller t_peak actually clips the MAH at earlier times
    new_mah_params = dk.DEFAULT_MAH_PARAMS._replace(t_peak=1.0)
    dmhdt_new2, log_mah_new2 = dk._diffmah_kern(new_mah_params, tarr, logt0)
    assert np.all(log_mah_new2 <= log_mah_new)
    assert np.any(log_mah_new2 < log_mah_new)

    msk_t = (tarr > 1.0) & (tarr < 2.0)
    assert np.all(log_mah_new2[msk_t] != log_mah_new[msk_t])


def test_param_u_param_names_propagate_properly():
    gen = zip(dk.DEFAULT_MAH_U_PARAMS._fields, dk.DEFAULT_MAH_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = dk.get_bounded_mah_params(dk.DEFAULT_MAH_U_PARAMS)
    assert set(inferred_default_params._fields) == set(dk.DEFAULT_MAH_PARAMS._fields)

    inferred_default_u_params = dk.get_unbounded_mah_params(dk.DEFAULT_MAH_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        dk.DEFAULT_MAH_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        dk.get_bounded_mah_params(dk.DEFAULT_MAH_PARAMS)
        raise NameError("get_bounded_mah_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        dk.get_unbounded_mah_params(dk.DEFAULT_MAH_U_PARAMS)
        raise NameError("get_unbounded_deltapop_params should not accept u_params")
    except AttributeError:
        pass


def test_param_recovery_from_u_params():
    mah_params = dk.DEFAULT_MAH_PARAMS
    mah_u_params = dk.get_unbounded_mah_params(mah_params)
    assert np.allclose(mah_u_params, dk.DEFAULT_MAH_U_PARAMS, rtol=1e-3)

    mah_params2 = dk.get_bounded_mah_params(mah_u_params)
    assert np.allclose(mah_params, mah_params2, rtol=1e-3)


def test_diffmah_kern_u_params():
    tarr = np.linspace(0.01, 13.8, 1_000)
    logt0 = np.log10(tarr[-1])
    mah_params = dk.DEFAULT_MAH_PARAMS._replace(t_peak=4.0)
    dmhdt, log_mah = dk._diffmah_kern(mah_params, tarr, logt0)
    mah_u_params = dk.get_unbounded_mah_params(mah_params)
    dmhdt2, log_mah2 = dk._diffmah_kern_u_params(mah_u_params, tarr, logt0)
    assert np.allclose(dmhdt, dmhdt2, rtol=1e-3)
    assert np.allclose(log_mah, log_mah2, rtol=1e-3)


def test_dmhdt_kern_u_params():
    tarr = np.linspace(0.01, 13.8, 1_000)
    logt0 = np.log10(tarr[-1])
    mah_params = dk.DEFAULT_MAH_PARAMS._replace(t_peak=4.0)
    dmhdt = dk._dmhdt_kern(mah_params, tarr, logt0)

    mah_u_params = dk.get_unbounded_mah_params(mah_params)
    dmhdt2 = dk._dmhdt_kern_u_params(mah_u_params, tarr, logt0)

    assert np.allclose(dmhdt, dmhdt2, rtol=1e-3)


def test_log_mah_kern_u_params():
    tarr = np.linspace(0.01, 13.8, 1_000)
    logt0 = np.log10(tarr[-1])
    mah_params = dk.DEFAULT_MAH_PARAMS._replace(t_peak=4.0)
    log_mah = dk._log_mah_kern(mah_params, tarr, logt0)

    mah_u_params = dk.get_unbounded_mah_params(mah_params)
    log_mah2 = dk._log_mah_kern_u_params(mah_u_params, tarr, logt0)

    assert np.allclose(log_mah, log_mah2, rtol=1e-3)


def test_logmh_at_t_obs_singlehalo():
    mah_params = dk.DEFAULT_MAH_PARAMS
    lgt0 = 1.14

    # single halo
    t_obs = 13.0
    logmh_at_t_obs = dk.logmh_at_t_obs(mah_params, t_obs, lgt0)
    assert logmh_at_t_obs.shape == ()

    t_table = np.linspace(0.1, 13.8, 500)

    log_mah = dk.mah_singlehalo(mah_params, t_table, lgt0)[1]
    logmh_at_t_obs_interp = np.interp(t_obs, t_table, log_mah)
    assert np.allclose(logmh_at_t_obs, logmh_at_t_obs_interp, rtol=0.01)


def test_logmh_at_t_obs_halopop():
    ran_key = jran.key(0)
    lgt0 = 1.14

    t_table = np.linspace(0.1, 13.8, 500)

    # halo population
    n_halos = 500
    ZZ = np.zeros(n_halos)
    mah_params = dk.DEFAULT_MAH_PARAMS._make([ZZ + x for x in dk.DEFAULT_MAH_PARAMS])
    t_obs_halopop = jran.uniform(ran_key, minval=3, maxval=13, shape=(n_halos,))
    logmh_at_t_obs = dk.logmh_at_t_obs(mah_params, t_obs_halopop, lgt0)

    log_mah_halopop = dk.mah_halopop(mah_params, t_table, lgt0)[1]

    interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))
    logmh_at_t_obs_interp = interp_vmap(t_obs_halopop, t_table, log_mah_halopop)
    assert np.allclose(logmh_at_t_obs, logmh_at_t_obs_interp, rtol=0.01)
