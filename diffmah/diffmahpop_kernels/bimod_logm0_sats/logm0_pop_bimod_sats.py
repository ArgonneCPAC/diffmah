"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import (
    logm0_c0_early,
    logm0_c0_late,
    logm0_c1_early,
    logm0_c1_late,
    logm0_pop_early,
    logm0_pop_late,
)

DEFAULT_LOGM0_PDICT = OrderedDict()
DEFAULT_LOGM0_PDICT.update(logm0_pop_early.DEFAULT_LOGM0_PDICT)
DEFAULT_LOGM0_PDICT.update(logm0_pop_late.DEFAULT_LOGM0_PDICT)

LGM0Pop_Params = namedtuple("LGM0Pop_Params", DEFAULT_LOGM0_PDICT.keys())
DEFAULT_LOGM0POP_PARAMS = LGM0Pop_Params(**DEFAULT_LOGM0_PDICT)

_UPNAMES = ["u_" + key for key in LGM0Pop_Params._fields]
LGM0Pop_UParams = namedtuple("LGM0Pop_UParams", _UPNAMES)

DEFAULT_LOGM0_BOUNDS_DICT = OrderedDict()
DEFAULT_LOGM0_BOUNDS_DICT.update(logm0_pop_early.DEFAULT_LOGM0_BOUNDS_DICT)
DEFAULT_LOGM0_BOUNDS_DICT.update(logm0_pop_late.DEFAULT_LOGM0_BOUNDS_DICT)
LGM0POP_BOUNDS = LGM0Pop_Params(**DEFAULT_LOGM0_BOUNDS_DICT)


@jjit
def get_bounded_m0pop_params(u_params):
    c0_early_u_params = [getattr(u_params, key) for key in logm0_c0_early._C0_UPNAMES]
    c0_early_u_params = logm0_c0_early.LGM0Pop_C0_UParams(*c0_early_u_params)
    c0_early_params = logm0_c0_early.get_bounded_c0_params(c0_early_u_params)

    c1_early_u_params = [getattr(u_params, key) for key in logm0_c1_early._C1_UPNAMES]
    c1_early_u_params = logm0_c1_early.LGM0Pop_C1_UParams(*c1_early_u_params)
    c1_early_params = logm0_c1_early.get_bounded_c1_params(c1_early_u_params)

    c0_late_u_params = [getattr(u_params, key) for key in logm0_c0_late._C0_UPNAMES]
    c0_late_u_params = logm0_c0_late.LGM0Pop_C0_UParams(*c0_late_u_params)
    c0_late_params = logm0_c0_late.get_bounded_c0_params(c0_late_u_params)

    c1_late_u_params = [getattr(u_params, key) for key in logm0_c1_late._C1_UPNAMES]
    c1_late_u_params = logm0_c1_late.LGM0Pop_C1_UParams(*c1_late_u_params)
    c1_late_params = logm0_c1_late.get_bounded_c1_params(c1_late_u_params)

    params = LGM0Pop_Params(
        *c0_early_params, *c1_early_params, *c0_late_params, *c1_late_params
    )
    return params


@jjit
def get_unbounded_m0pop_params(params):
    c0_early_pnames = logm0_c0_early.LGM0Pop_C0_Params._fields
    c1_early_pnames = logm0_c1_early.LGM0Pop_C1_Params._fields

    c0_late_pnames = logm0_c0_late.LGM0Pop_C0_Params._fields
    c1_late_pnames = logm0_c1_late.LGM0Pop_C1_Params._fields

    c0_early_params = [getattr(params, key) for key in c0_early_pnames]
    c0_early_params = logm0_c0_early.LGM0Pop_C0_Params(*c0_early_params)
    c0_early_u_params = logm0_c0_early.get_unbounded_c0_params(c0_early_params)

    c1_early_params = [getattr(params, key) for key in c1_early_pnames]
    c1_early_params = logm0_c1_early.LGM0Pop_C1_Params(*c1_early_params)
    c1_early_u_params = logm0_c1_early.get_unbounded_c1_params(c1_early_params)

    c0_late_params = [getattr(params, key) for key in c0_late_pnames]
    c0_late_params = logm0_c0_late.LGM0Pop_C0_Params(*c0_late_params)
    c0_late_u_params = logm0_c0_late.get_unbounded_c0_params(c0_late_params)

    c1_late_params = [getattr(params, key) for key in c1_late_pnames]
    c1_late_params = logm0_c1_late.LGM0Pop_C1_Params(*c1_late_params)
    c1_late_u_params = logm0_c1_late.get_unbounded_c1_params(c1_late_params)

    u_params = LGM0Pop_UParams(
        *c0_early_u_params, *c1_early_u_params, *c0_late_u_params, *c1_late_u_params
    )
    return u_params


@jjit
def _pred_logm0_kern_early(logm0_params, lgm_obs, t_obs, t_peak):
    return logm0_pop_early._pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)


@jjit
def _pred_logm0_kern_late(logm0_params, lgm_obs, t_obs, t_peak):
    return logm0_pop_late._pred_logm0_kern(logm0_params, lgm_obs, t_obs, t_peak)


DEFAULT_LOGM0POP_U_PARAMS = LGM0Pop_UParams(
    *get_unbounded_m0pop_params(DEFAULT_LOGM0POP_PARAMS)
)
