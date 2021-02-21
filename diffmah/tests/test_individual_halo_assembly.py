"""
"""
from jax import random as jran
from jax import numpy as jnp
from ..individual_halo_assembly import _get_params_from_u_params
from ..individual_halo_assembly import _get_u_params_from_params
from ..individual_halo_assembly import _MAH_PARS, _MAH_BOUNDS


def test_bounded_params_are_correctly_bounded():
    n_tests = 100
    ran_key = jran.PRNGKey(43)
    for seed in range(n_tests):
        old_key, ran_key = jran.split(ran_key)
        up = 100 * (2 * jran.uniform(ran_key, (4,)) - 1)
        p = _get_params_from_u_params(*up)
        x0, k, early, late = p
        assert _MAH_BOUNDS["mah_x0"][0] < float(x0) < _MAH_BOUNDS["mah_x0"][1]
        assert _MAH_BOUNDS["mah_k"][0] < float(k) < _MAH_BOUNDS["mah_k"][1]
        assert _MAH_BOUNDS["mah_early"][0] < float(early) < _MAH_BOUNDS["mah_early"][1]
        assert 0 < float(late) < float(early)


def test_bounded_params_properly_inverts():
    n_tests = 100
    ran_key = jran.PRNGKey(43)
    for seed in range(n_tests):
        old_key, ran_key = jran.split(ran_key)
        up = 100 * (2 * jran.uniform(ran_key, (4,)) - 1)
        u_x0, u_k, u_early, u_dy = up
        p = _get_params_from_u_params(*up)
        x0, k, early, late = p

        up2 = _get_u_params_from_params(*p)
        u_x02, u_k2, u_early2, u_dy2 = up2

        atol = 0.01
        assert jnp.allclose(u_x0, u_x02, atol=atol)
        assert jnp.allclose(u_k, u_k2, atol=atol)
        assert jnp.allclose(u_early, u_early, atol=atol)
        assert jnp.allclose(u_dy, u_dy2, atol=atol)

        p2 = _get_params_from_u_params(*up2)
        x02, k2, early2, late2 = p2
        assert jnp.allclose(x0, x02, atol=atol)
        assert jnp.allclose(k, k2, atol=atol)
        assert jnp.allclose(early, early2, atol=atol)
        assert jnp.allclose(late, late2, atol=atol)


def test_default_params_are_correctly_bounded():
    for key, bound in _MAH_BOUNDS.items():
        lo, hi = bound
        assert float(lo) < float(_MAH_PARS[key]) < float(hi)
