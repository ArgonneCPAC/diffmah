"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import early_index_alpha, early_index_beta

EARLY_INDEX_PDICT = OrderedDict()
EARLY_INDEX_PDICT.update(early_index_alpha.EARLY_INDEX_PDICT)
EARLY_INDEX_PDICT.update(early_index_beta.EARLY_INDEX_PDICT)

EarlyIndex_Params = namedtuple("EarlyIndex_Params", EARLY_INDEX_PDICT.keys())
DEFAULT_EARLY_INDEX_PARAMS = EarlyIndex_Params(**EARLY_INDEX_PDICT)

_UPNAMES = ["u_" + key for key in EarlyIndex_Params._fields]
EarlyIndex_UParams = namedtuple("EarlyIndex_UParams", _UPNAMES)

EARLY_INDEX_BOUNDS_PDICT = OrderedDict()
EARLY_INDEX_BOUNDS_PDICT.update(early_index_alpha.EARLY_INDEX_BOUNDS_PDICT)
EARLY_INDEX_BOUNDS_PDICT.update(early_index_beta.EARLY_INDEX_BOUNDS_PDICT)
EARLY_INDEX_PBOUNDS = EarlyIndex_Params(**EARLY_INDEX_BOUNDS_PDICT)


@jjit
def _pred_early_index_early(params, lgm_obs, t_obs, t_peak):
    return early_index_alpha._pred_early_index_kern(params, lgm_obs, t_obs, t_peak)


@jjit
def _pred_early_index_late(params, lgm_obs, t_obs, t_peak):
    return early_index_beta._pred_early_index_kern(params, lgm_obs, t_obs, t_peak)


@jjit
def get_bounded_early_index_params(u_params):
    early_u_params = [
        getattr(u_params, key)
        for key in early_index_alpha.DEFAULT_EARLY_INDEX_U_PARAMS._fields
    ]
    early_u_params = early_index_alpha.EarlyIndex_UParams(*early_u_params)
    early_params = early_index_alpha.get_bounded_early_index_params(early_u_params)

    late_u_params = [
        getattr(u_params, key)
        for key in early_index_beta.DEFAULT_EARLY_INDEX_U_PARAMS._fields
    ]
    late_u_params = early_index_beta.EarlyIndex_UParams(*late_u_params)
    late_params = early_index_beta.get_bounded_early_index_params(late_u_params)

    params = EarlyIndex_Params(*early_params, *late_params)
    return params


@jjit
def get_unbounded_early_index_params(params):
    early_pnames = early_index_alpha.EarlyIndex_Params._fields
    late_pnames = early_index_beta.EarlyIndex_Params._fields

    early_params = [getattr(params, key) for key in early_pnames]
    early_params = early_index_alpha.EarlyIndex_Params(*early_params)
    early_u_params = early_index_alpha.get_unbounded_early_index_params(early_params)

    late_params = [getattr(params, key) for key in late_pnames]
    late_params = early_index_beta.EarlyIndex_Params(*late_params)
    late_u_params = early_index_beta.get_unbounded_early_index_params(late_params)

    u_params = EarlyIndex_UParams(*early_u_params, *late_u_params)
    return u_params


DEFAULT_EARLY_INDEX_U_PARAMS = EarlyIndex_UParams(
    *get_unbounded_early_index_params(DEFAULT_EARLY_INDEX_PARAMS)
)
