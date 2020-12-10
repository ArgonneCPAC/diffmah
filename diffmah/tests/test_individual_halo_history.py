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
        lge, u_dy = jran.uniform(ran_key, (2,))
        lge2, u_dy2 = _get_unbounded_params(*_get_bounded_params(lge, u_dy))
        assert jnp.allclose(lge, lge2)
        assert jnp.allclose(u_dy, u_dy2)
