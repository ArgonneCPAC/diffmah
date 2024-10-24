"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap
from jax.scipy.stats import truncnorm

from ...bfgs_wrapper import bfgs_adam_fallback
from ...utils import _inverse_sigmoid, _sigmoid

K_BOUNDING = 0.1
X_MIN, X_MAX = 0.1, 1.0
NPTS_CDF_TABLE = 50
EPS = 1e-4
X_CDF_TABLE = jnp.linspace(X_MIN + EPS, X_MAX - EPS, NPTS_CDF_TABLE)

DEFAULT_UTP_PDICT = OrderedDict(utp_loc=0.5, utp_scale=0.5)


UTP_PBOUNDS_PDICT = OrderedDict(
    utp_loc=(X_MIN + EPS, X_MAX - EPS), utp_scale=(0.0 + EPS, 1.0)
)


UTP_Params = namedtuple("UTP_Params", DEFAULT_UTP_PDICT.keys())
_UTP_UPNAMES = ["u_" + key for key in UTP_Params._fields]
UTP_UParams = namedtuple("UTP_UParams", _UTP_UPNAMES)


DEFAULT_UTP_PARAMS = UTP_Params(**DEFAULT_UTP_PDICT)
UTP_PBOUNDS = UTP_Params(**UTP_PBOUNDS_PDICT)


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d * d)


@jjit
def mc_tp_pdf_singlesat(ran_key, params):
    loc, scale = params
    x_min = jnp.max(jnp.array((loc - 4 * scale, X_MIN + EPS)))
    x_max = jnp.min(jnp.array((loc + 4 * scale, X_MAX - EPS)))
    x_cdf_table = jnp.linspace(x_min, x_max, NPTS_CDF_TABLE)
    cdf_table = _tp_cdf_kern(x_cdf_table, params)
    u_ran = jran.uniform(ran_key, minval=0.0, maxval=1.0, shape=())
    x = jnp.interp(u_ran, cdf_table, x_cdf_table)
    return x


_mc_tp_pdf_satpop = jjit(vmap(mc_tp_pdf_singlesat, in_axes=(0, 0)))


@jjit
def mc_tp_pdf_satpop(ran_key, params):
    n_sats = params.utp_loc.size
    ran_keys = jran.split(ran_key, n_sats)
    return _mc_tp_pdf_satpop(ran_keys, params)


@jjit
def _get_a_b_loc_scale(params, x_min=X_MIN, x_max=X_MAX):
    loc, scale = params
    a = (x_min - loc) / scale
    b = (x_max - loc) / scale
    return a, b, loc, scale


@jjit
def _tp_cdf_kern(x, params, x_min=X_MIN, x_max=X_MAX):
    a, b, loc, scale = _get_a_b_loc_scale(params, x_min, x_max)
    cdf = truncnorm.cdf(x, a, b, loc, scale)
    return cdf


@jjit
def _tp_pdf_kern(x, params, x_min=X_MIN, x_max=X_MAX):
    a, b, loc, scale = _get_a_b_loc_scale(params, x_min, x_max)
    pdf = truncnorm.pdf(x, a, b, loc, scale)
    return pdf


@jjit
def _loss_kern(u_params, loss_data):
    u_params = UTP_UParams(*u_params)
    params = get_bounded_utp_params(u_params)
    x_target, pdf_target = loss_data
    pdf_pred = _tp_pdf_kern(x_target, params)
    return _mse(pdf_pred, pdf_target)


loss_and_grads_kern = jjit(value_and_grad(_loss_kern))


@jjit
def _get_bounded_utp_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_utp_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_utp_params_kern = jjit(vmap(_get_bounded_utp_param, in_axes=_C))
_get_unbounded_utp_params_kern = jjit(vmap(_get_unbounded_utp_param, in_axes=_C))


@jjit
def get_bounded_utp_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _UTP_UPNAMES])
    params = _get_bounded_utp_params_kern(jnp.array(u_params), jnp.array(UTP_PBOUNDS))
    utp_params = UTP_Params(*params)
    return utp_params


@jjit
def get_unbounded_utp_params(params):
    params = jnp.array([getattr(params, pname) for pname in DEFAULT_UTP_PARAMS._fields])
    u_params = _get_unbounded_utp_params_kern(jnp.array(params), jnp.array(UTP_PBOUNDS))
    utp_u_params = UTP_UParams(*u_params)
    return utp_u_params


DEFAULT_UTP_U_PARAMS = UTP_UParams(*get_unbounded_utp_params(DEFAULT_UTP_PARAMS))


def t_peak_fitter(x_target, pdf_target):
    loss_data = x_target, pdf_target
    args = (loss_and_grads_kern, DEFAULT_UTP_U_PARAMS, loss_data)

    u_p_best, loss_best, fit_terminates, code_used = bfgs_adam_fallback(*args)
    u_p_best = UTP_UParams(*u_p_best)
    p_best = get_bounded_utp_params(u_p_best)
    return p_best, loss_best, fit_terminates, code_used
