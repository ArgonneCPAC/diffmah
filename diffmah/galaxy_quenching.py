"""Model for the permanent quenching of individual galaxies."""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from .utils import _jax_tw_cuml_kern, jax_sigmoid, jax_inverse_sigmoid


DEFAULT_Q_PARAMS = OrderedDict(u_lg_qt=0.0, u_lg_qs=0.0, u_lg_dq=1.0)
Q_PARAM_BOUNDS = OrderedDict(u_lg_qt=(0.5, 1.25), u_lg_qs=(-3, 0.0), u_lg_dq=(-3, -0.5))


@jjit
def _quenching_kern(lgt, u_lg_qt, u_lg_qs, u_lg_dq):
    lg_qt, qs, lg_dq = _get_bounded_params(u_lg_qt, u_lg_qs, u_lg_dq)
    return 10 ** _jax_tw_qfunc_kern(lgt, lg_qt, qs, lg_dq)


@jjit
def _get_bounded_params(u_lg_qt, u_lg_qs, u_lg_dq):
    lg_qt = jax_sigmoid(u_lg_qt, 0, 0.1, *Q_PARAM_BOUNDS["u_lg_qt"])
    qs = 10 ** jax_sigmoid(u_lg_qs, 0, 0.1, *Q_PARAM_BOUNDS["u_lg_qs"])
    lg_dq = jax_sigmoid(u_lg_dq, 0, 0.1, *Q_PARAM_BOUNDS["u_lg_dq"])
    return lg_qt, qs, lg_dq


@jjit
def _get_unbounded_params(lg_qt, qs, lg_dq):
    u_lg_qt = jax_inverse_sigmoid(lg_qt, 0, 0.1, *Q_PARAM_BOUNDS["u_lg_qt"])
    u_lg_qs = jax_inverse_sigmoid(jnp.log10(qs), 0, 0.1, *Q_PARAM_BOUNDS["u_lg_qs"])
    u_lg_dq = jax_inverse_sigmoid(lg_dq, 0, 0.1, *Q_PARAM_BOUNDS["u_lg_dq"])
    return u_lg_qt, u_lg_qs, u_lg_dq


@jjit
def _jax_tw_qfunc_kern(lgt, lgqt, tw_h, lgdq):
    tw_m = 3 * tw_h + lgqt
    log_sfr_drop = lgdq * _jax_tw_cuml_kern(lgt, tw_m, tw_h)
    return log_sfr_drop


_quenching_kern_vmap = jvmap(_quenching_kern, in_axes=(0, None, None, None))
