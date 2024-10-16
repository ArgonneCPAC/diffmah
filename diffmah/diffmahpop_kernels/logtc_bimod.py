"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import logtc_early, logtc_late

LOGTC_PDICT = OrderedDict()
LOGTC_PDICT.update(logtc_early.LOGTC_PDICT)
LOGTC_PDICT.update(logtc_late.LOGTC_PDICT)

Logtc_Params = namedtuple("Logtc_Params", LOGTC_PDICT.keys())
DEFAULT_LOGTC_PARAMS = Logtc_Params(**LOGTC_PDICT)

_UPNAMES = ["u_" + key for key in Logtc_Params._fields]
Logtc_UParams = namedtuple("Logtc_UParams", _UPNAMES)

LOGTC_BOUNDS_PDICT = OrderedDict()
LOGTC_BOUNDS_PDICT.update(logtc_early.LOGTC_BOUNDS_PDICT)
LOGTC_BOUNDS_PDICT.update(logtc_late.LOGTC_BOUNDS_PDICT)
LOGTC_PBOUNDS = Logtc_Params(**LOGTC_BOUNDS_PDICT)


@jjit
def get_bounded_logtc_params(u_params):
    early_u_params = [
        getattr(u_params, key) for key in logtc_early.DEFAULT_LOGTC_U_PARAMS._fields
    ]
    early_u_params = logtc_early.Logtc_UParams(*early_u_params)
    early_params = logtc_early.get_bounded_logtc_params(early_u_params)

    late_u_params = [
        getattr(u_params, key) for key in logtc_late.DEFAULT_LOGTC_U_PARAMS._fields
    ]
    late_u_params = logtc_late.Logtc_UParams(*late_u_params)
    late_params = logtc_late.get_bounded_logtc_params(late_u_params)

    params = Logtc_Params(*early_params, *late_params)
    return params


@jjit
def get_unbounded_logtc_params(params):
    early_pnames = logtc_early.Logtc_Params._fields
    late_pnames = logtc_late.Logtc_Params._fields

    early_params = [getattr(params, key) for key in early_pnames]
    early_params = logtc_early.Logtc_Params(*early_params)
    early_u_params = logtc_early.get_unbounded_logtc_params(early_params)

    late_params = [getattr(params, key) for key in late_pnames]
    late_params = logtc_late.Logtc_Params(*late_params)
    late_u_params = logtc_late.get_unbounded_logtc_params(late_params)

    u_params = Logtc_UParams(*early_u_params, *late_u_params)
    return u_params


DEFAULT_LOGTC_U_PARAMS = Logtc_UParams(
    *get_unbounded_logtc_params(DEFAULT_LOGTC_PARAMS)
)
