"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import logm0_c0_early, logm0_c1_early

DEFAULT_LOGM0_PDICT = OrderedDict()
DEFAULT_LOGM0_PDICT.update(logm0_c0_early.DEFAULT_LGM0POP_C0_PDICT)
DEFAULT_LOGM0_PDICT.update(logm0_c1_early.DEFAULT_LGM0POP_C1_PDICT)

LGM0Pop_Params = namedtuple("LGM0Pop_Params", DEFAULT_LOGM0_PDICT.keys())
DEFAULT_LOGM0POP_PARAMS = LGM0Pop_Params(**DEFAULT_LOGM0_PDICT)

_UPNAMES = ["u_" + key for key in LGM0Pop_Params._fields]
LGM0Pop_UParams = namedtuple("LGM0Pop_UParams", _UPNAMES)

DEFAULT_LOGM0_BOUNDS_DICT = OrderedDict()
DEFAULT_LOGM0_BOUNDS_DICT.update(logm0_c0_early.LGM0POP_C0_BOUNDS_DICT)
DEFAULT_LOGM0_BOUNDS_DICT.update(logm0_c1_early.LGM0POP_C1_BOUNDS_DICT)
LGM0POP_BOUNDS = LGM0Pop_Params(**DEFAULT_LOGM0_BOUNDS_DICT)

TP_LGMP = 12.0


@jjit
def _pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak):
    c0 = logm0_c0_early._pred_c0_kern(logm0_params, t_obs, t_peak)
    c1 = logm0_c1_early._pred_c1_kern(logm0_params, t_obs, t_peak)
    delta_lgm = c0 + c1 * (lgm_obs - TP_LGMP)
    return lgm_obs + delta_lgm


@jjit
def get_bounded_m0pop_params(u_params):
    c0_u_params = [getattr(u_params, key) for key in logm0_c0_early._C0_UPNAMES]
    c0_u_params = logm0_c0_early.LGM0Pop_C0_UParams(*c0_u_params)
    c0_params = logm0_c0_early.get_bounded_c0_params(c0_u_params)

    c1_u_params = [getattr(u_params, key) for key in logm0_c1_early._C1_UPNAMES]
    c1_u_params = logm0_c1_early.LGM0Pop_C1_UParams(*c1_u_params)
    c1_params = logm0_c1_early.get_bounded_c1_params(c1_u_params)

    params = LGM0Pop_Params(*c0_params, *c1_params)
    return params


@jjit
def get_unbounded_m0pop_params(params):
    c0_pnames = logm0_c0_early.LGM0Pop_C0_Params._fields
    c1_pnames = logm0_c1_early.LGM0Pop_C1_Params._fields

    c0_params = [getattr(params, key) for key in c0_pnames]
    c0_params = logm0_c0_early.LGM0Pop_C0_Params(*c0_params)
    c0_u_params = logm0_c0_early.get_unbounded_c0_params(c0_params)

    c1_params = [getattr(params, key) for key in c1_pnames]
    c1_params = logm0_c1_early.LGM0Pop_C1_Params(*c1_params)
    c1_u_params = logm0_c1_early.get_unbounded_c1_params(c1_params)

    u_params = LGM0Pop_UParams(*c0_u_params, *c1_u_params)
    return u_params


DEFAULT_LOGM0POP_U_PARAMS = LGM0Pop_UParams(
    *get_unbounded_m0pop_params(DEFAULT_LOGM0POP_PARAMS)
)
