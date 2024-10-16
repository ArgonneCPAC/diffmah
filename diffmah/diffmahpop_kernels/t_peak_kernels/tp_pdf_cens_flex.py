"""Sigmoid model of the CDF P_cen(t<t_peak | lgm_obs)

"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad, vmap

from ...utils import _inverse_sigmoid, _sigmoid

YLO = 0.0
YHI = 1.0
K = 5.0
UTP_LGM_X0_K = 1.0
UTP_T_OBS_X0_K = 2.0


UTP_MIN = 0.05

DEFAULT_TPCENS_PDICT = OrderedDict(
    cen_tp_lgm_x0_x0=11.010,
    cen_tp_t_obs_x0_x0=3.798,
    cen_tp_x0_ylo_early=0.99,
    cen_tp_x0_yhi_early=1.750,
    cen_tp_x0_ylo_late=0.690,
    cen_tp_x0_yhi_late=1.351,
)
TPCens_Params = namedtuple("TPCens_Params", DEFAULT_TPCENS_PDICT.keys())
DEFAULT_TPCENS_PARAMS = TPCens_Params(**DEFAULT_TPCENS_PDICT)

CEN_TP_PDF_BOUNDS_DICT = OrderedDict(
    cen_tp_lgm_x0_x0=(11.0, 14.0),
    cen_tp_t_obs_x0_x0=(3.0, 10.0),
    cen_tp_x0_ylo_early=(0.5, 1.0),
    cen_tp_x0_yhi_early=(1.0, 1.8),
    cen_tp_x0_ylo_late=(0.5, 1.0),
    cen_tp_x0_yhi_late=(1.0, 1.8),
)
TPCENS_PBOUNDS = TPCens_Params(**CEN_TP_PDF_BOUNDS_DICT)

_TPCens_UPNAMES = ["u_" + key for key in TPCens_Params._fields]
TPCens_UParams = namedtuple("TPCens_UParams", _TPCens_UPNAMES)

K_BOUNDING = 0.1

UTP_TABLE = jnp.linspace(0.0, 1.0, 200)


@jjit
def mc_tpeak_singlecen(params, lgm_obs, t_obs, ran_key, t_0):
    utp = mc_utp_singlecen(params, lgm_obs, t_obs, ran_key)
    return utp * t_0


_CP = (None, 0, 0, 0, None)
mc_t_peak_cenpop_kern = jjit(vmap(mc_tpeak_singlecen, in_axes=_CP))


@jjit
def mc_t_peak_cenpop(params, lgm_obs, t_obs, ran_key, t_0):
    ran_keys = jran.split(ran_key, lgm_obs.size)
    return mc_t_peak_cenpop_kern(params, lgm_obs, t_obs, ran_keys, t_0)


@jjit
def mc_utp_singlecen(params, lgm_obs, t_obs, ran_key):
    fp_table = _frac_peaked_vs_lgm(params, UTP_TABLE, lgm_obs, t_obs)
    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=())
    utp = jnp.interp(uran, fp_table, UTP_TABLE)
    utp = jnp.clip(utp, UTP_MIN)
    return utp


@jjit
def _frac_peaked_kern(utp, utp_x0):
    return _sigmoid(utp, utp_x0, K, YLO, YHI)


@jjit
def _get_utp_x0_from_lgm(params, lgm, t_obs):
    utp_x0_early = _sigmoid(
        lgm,
        params.cen_tp_lgm_x0_x0,
        UTP_LGM_X0_K,
        params.cen_tp_x0_ylo_early,
        params.cen_tp_x0_yhi_early,
    )
    utp_x0_late = _sigmoid(
        lgm,
        params.cen_tp_lgm_x0_x0,
        UTP_LGM_X0_K,
        params.cen_tp_x0_ylo_late,
        params.cen_tp_x0_yhi_late,
    )
    utp_x0 = _sigmoid(
        t_obs, params.cen_tp_t_obs_x0_x0, UTP_T_OBS_X0_K, utp_x0_early, utp_x0_late
    )
    return utp_x0


@jjit
def _frac_peaked_vs_lgm(params, utp, lgm, t_obs):
    utp_x0 = _get_utp_x0_from_lgm(params, lgm, t_obs)
    return _sigmoid(utp, utp_x0, K, YLO, YHI)


@jjit
def _mse(x, y):
    d = y - x
    return jnp.mean(d**2)


@jjit
def _mae(x, y):
    d = y - x
    return jnp.abs(d)


@jjit
def _loss_kern_single_mass(params, utp_target, y_target, lgm_obs, t_obs):
    y_pred = _frac_peaked_vs_lgm(params, utp_target, lgm_obs, t_obs)
    return _mae(y_pred, y_target)


_loss_kern_single_mass_vmap = jjit(
    vmap(_loss_kern_single_mass, in_axes=(None, 0, 0, 0, 0))
)


@jjit
def loss_multimass(params, loss_data):
    utp_targets, y_targets, lgm_obs_arr, t_obs_arr = loss_data
    losses = _loss_kern_single_mass_vmap(
        params, utp_targets, y_targets, lgm_obs_arr, t_obs_arr
    )
    return jnp.mean(losses)


loss_and_grad_multimass = jjit(value_and_grad(loss_multimass))


@jjit
def _loss(params, loss_data):
    (utp_x0,) = params
    utp_target, y_target = loss_data
    y_pred = _frac_peaked_kern(utp_target, utp_x0)
    return _mse(y_pred, y_target)


loss_and_grad = jjit(value_and_grad(_loss))


@jjit
def _get_bounded_tp_cens_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, K_BOUNDING, lo, hi)


@jjit
def _get_unbounded_tp_cens_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, K_BOUNDING, lo, hi)


_C = (0, 0)
_get_bounded_tp_cens_params_kern = jjit(vmap(_get_bounded_tp_cens_param, in_axes=_C))
_get_unbounded_tp_cens_params_kern = jjit(
    vmap(_get_unbounded_tp_cens_param, in_axes=_C)
)


@jjit
def get_bounded_tp_cens_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _TPCens_UPNAMES])
    params = _get_bounded_tp_cens_params_kern(
        jnp.array(u_params), jnp.array(TPCENS_PBOUNDS)
    )
    params = TPCens_Params(*params)
    return params


@jjit
def get_unbounded_tp_cens_params(params):
    params = jnp.array([getattr(params, pname) for pname in TPCens_Params._fields])
    u_params = _get_unbounded_tp_cens_params_kern(
        jnp.array(params), jnp.array(TPCENS_PBOUNDS)
    )
    u_params = TPCens_UParams(*u_params)
    return u_params


DEFAULT_TPCENS_U_PARAMS = TPCens_UParams(
    *get_unbounded_tp_cens_params(DEFAULT_TPCENS_PARAMS)
)
