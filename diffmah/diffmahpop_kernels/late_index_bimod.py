"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import late_index_alpha, late_index_beta

LATE_INDEX_PDICT = OrderedDict()
LATE_INDEX_PDICT.update(late_index_alpha.LATE_INDEX_PDICT)
LATE_INDEX_PDICT.update(late_index_beta.LATE_INDEX_PDICT)

LateIndex_Params = namedtuple("LateIndex_Params", LATE_INDEX_PDICT.keys())
DEFAULT_LATE_INDEX_PARAMS = LateIndex_Params(**LATE_INDEX_PDICT)

_UPNAMES = ["u_" + key for key in LateIndex_Params._fields]
LateIndex_UParams = namedtuple("LateIndex_UParams", _UPNAMES)

LATE_INDEX_BOUNDS_PDICT = OrderedDict()
LATE_INDEX_BOUNDS_PDICT.update(late_index_alpha.LATE_INDEX_BOUNDS_PDICT)
LATE_INDEX_BOUNDS_PDICT.update(late_index_beta.LATE_INDEX_BOUNDS_PDICT)
LATE_INDEX_PBOUNDS = LateIndex_Params(**LATE_INDEX_BOUNDS_PDICT)


@jjit
def _pred_late_index_early(params, lgm_obs):
    return late_index_alpha._pred_late_index_kern(params, lgm_obs)


@jjit
def _pred_late_index_late(params, lgm_obs):
    return late_index_beta._pred_late_index_kern(params, lgm_obs)


@jjit
def get_bounded_late_index_params(u_params):
    late_u_params = [
        getattr(u_params, key)
        for key in late_index_alpha.DEFAULT_LATE_INDEX_U_PARAMS._fields
    ]
    late_u_params = late_index_alpha.LateIndex_UParams(*late_u_params)
    late_params = late_index_alpha.get_bounded_late_index_params(late_u_params)

    late_u_params = [
        getattr(u_params, key)
        for key in late_index_beta.DEFAULT_LATE_INDEX_U_PARAMS._fields
    ]
    late_u_params = late_index_beta.LateIndex_UParams(*late_u_params)
    late_params = late_index_beta.get_bounded_late_index_params(late_u_params)

    params = LateIndex_Params(*late_params, *late_params)
    return params


@jjit
def get_unbounded_late_index_params(params):
    late_pnames = late_index_alpha.LateIndex_Params._fields
    late_pnames = late_index_beta.LateIndex_Params._fields

    late_params = [getattr(params, key) for key in late_pnames]
    late_params = late_index_alpha.LateIndex_Params(*late_params)
    late_u_params = late_index_alpha.get_unbounded_late_index_params(late_params)

    late_params = [getattr(params, key) for key in late_pnames]
    late_params = late_index_beta.LateIndex_Params(*late_params)
    late_u_params = late_index_beta.get_unbounded_late_index_params(late_params)

    u_params = LateIndex_UParams(*late_u_params, *late_u_params)
    return u_params


DEFAULT_LATE_INDEX_U_PARAMS = LateIndex_UParams(
    *get_unbounded_late_index_params(DEFAULT_LATE_INDEX_PARAMS)
)
