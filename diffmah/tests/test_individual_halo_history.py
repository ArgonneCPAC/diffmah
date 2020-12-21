"""
"""
from jax import random as jran
from jax import numpy as jnp
from ..individual_halo_history import _get_bounded_params, _get_unbounded_params


def test_bounded_params_inverts():
    n_tests = 10
    ran_key = jran.PRNGKey(43)
    for seed in range(n_tests):
        old_key, ran_key = jran.split(ran_key)
        p = jran.uniform(ran_key, (4,))
        x0, k, lge, u_dy = p
        p2 = _get_unbounded_params(*_get_bounded_params(*p))
        x02, k2, lge2, u_dy2 = p2
        atol = 1e-4
        assert jnp.allclose(x0, x02, atol=atol)
        assert jnp.allclose(k, k2, atol=atol)
        assert jnp.allclose(lge, lge2, atol=atol)
        assert jnp.allclose(u_dy, u_dy2, atol=atol)
