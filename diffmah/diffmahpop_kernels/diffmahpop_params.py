"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import covariance_kernels, early_index_pop, ftpt0_cens, late_index_pop, logtc_pop
from .logm0_kernels import logm0_pop
from .t_peak_kernels import tp_pdf_cens

DEFAULT_DIFFMAHPOP_PDICT = OrderedDict()
COMPONENT_PDICTS = (
    ftpt0_cens.DEFAULT_FTPT0_PDICT,
    tp_pdf_cens.DEFAULT_TPCENS_PDICT,
    logm0_pop.DEFAULT_LOGM0_PDICT,
    logtc_pop.LOGTC_PDICT,
    early_index_pop.EARLY_INDEX_PDICT,
    late_index_pop.LATE_INDEX_PDICT,
    covariance_kernels.DEFAULT_COV_PDICT,
)
for pdict in COMPONENT_PDICTS:
    DEFAULT_DIFFMAHPOP_PDICT.update(pdict)
DiffmahPop_Params = namedtuple("DiffmahPop_Params", DEFAULT_DIFFMAHPOP_PDICT.keys())
DEFAULT_DIFFMAHPOP_PARAMS = DiffmahPop_Params(**DEFAULT_DIFFMAHPOP_PDICT)


COMPONENT_U_PDICTS = (
    ftpt0_cens.DEFAULT_FTPT0_U_PARAMS._asdict(),
    tp_pdf_cens.DEFAULT_TPCENS_U_PARAMS._asdict(),
    logm0_pop.DEFAULT_LOGM0POP_U_PARAMS._asdict(),
    logtc_pop.DEFAULT_LOGTC_U_PARAMS._asdict(),
    early_index_pop.DEFAULT_EARLY_INDEX_U_PARAMS._asdict(),
    late_index_pop.DEFAULT_LATE_INDEX_U_PARAMS._asdict(),
    covariance_kernels.DEFAULT_COV_U_PARAMS._asdict(),
)
DEFAULT_DIFFMAHPOP_U_PDICT = OrderedDict()
for updict in COMPONENT_U_PDICTS:
    DEFAULT_DIFFMAHPOP_U_PDICT.update(updict)
DiffmahPop_UParams = namedtuple("DiffmahPop_UParams", DEFAULT_DIFFMAHPOP_U_PDICT.keys())
DEFAULT_DIFFMAHPOP_U_PARAMS = DiffmahPop_UParams(**DEFAULT_DIFFMAHPOP_U_PDICT)


@jjit
def get_component_model_params(diffmahpop_params):
    ftpt0_cens_params = ftpt0_cens.FTPT0_Params(
        *[getattr(diffmahpop_params, key) for key in ftpt0_cens.FTPT0_Params._fields]
    )
    tp_pdf_cens_params = tp_pdf_cens.TPCens_Params(
        *[getattr(diffmahpop_params, key) for key in tp_pdf_cens.TPCens_Params._fields]
    )
    logm0_params = logm0_pop.LGM0Pop_Params(
        *[getattr(diffmahpop_params, key) for key in logm0_pop.LGM0Pop_Params._fields]
    )
    logtc_params = logtc_pop.Logtc_Params(
        *[getattr(diffmahpop_params, key) for key in logtc_pop.Logtc_Params._fields]
    )
    early_index_params = early_index_pop.EarlyIndex_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in early_index_pop.EarlyIndex_Params._fields
        ]
    )
    late_index_params = late_index_pop.LateIndex_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in late_index_pop.LateIndex_Params._fields
        ]
    )
    cov_params = covariance_kernels.CovParams(
        *[
            getattr(diffmahpop_params, key)
            for key in covariance_kernels.CovParams._fields
        ]
    )
    return (
        ftpt0_cens_params,
        tp_pdf_cens_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
        cov_params,
    )


@jjit
def get_component_model_u_params(diffmahpop_u_params):
    ftpt0_cens_u_params = ftpt0_cens.FTPT0_UParams(
        *[getattr(diffmahpop_u_params, key) for key in ftpt0_cens.FTPT0_UParams._fields]
    )
    tp_pdf_cens_u_params = tp_pdf_cens.TPCens_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in tp_pdf_cens.TPCens_UParams._fields
        ]
    )
    logm0_u_params = logm0_pop.LGM0Pop_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in logm0_pop.LGM0Pop_UParams._fields
        ]
    )
    logtc_u_params = logtc_pop.Logtc_UParams(
        *[getattr(diffmahpop_u_params, key) for key in logtc_pop.Logtc_UParams._fields]
    )
    early_index_u_params = early_index_pop.EarlyIndex_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in early_index_pop.EarlyIndex_UParams._fields
        ]
    )
    late_index_u_params = late_index_pop.LateIndex_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in late_index_pop.LateIndex_UParams._fields
        ]
    )
    cov_u_params = covariance_kernels.CovUParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in covariance_kernels.CovUParams._fields
        ]
    )

    return (
        ftpt0_cens_u_params,
        tp_pdf_cens_u_params,
        logm0_u_params,
        logtc_u_params,
        early_index_u_params,
        late_index_u_params,
        cov_u_params,
    )


@jjit
def get_diffmahpop_params_from_u_params(diffmahpop_u_params):
    component_model_u_params = get_component_model_u_params(diffmahpop_u_params)
    ftpt0_u_params, tpc_u_params, logm0_u_params = component_model_u_params[:3]
    logtc_u_params = component_model_u_params[3]
    early_index_u_params, late_index_u_params = component_model_u_params[4:6]
    cov_u_params = component_model_u_params[6]

    ftpt0_cens_params = ftpt0_cens.get_bounded_ftpt0_params(ftpt0_u_params)
    tpc_params = tp_pdf_cens.get_bounded_tp_cens_params(tpc_u_params)
    logm0_params = logm0_pop.get_bounded_m0pop_params(logm0_u_params)
    logtc_params = logtc_pop.get_bounded_logtc_params(logtc_u_params)
    early_index_params = early_index_pop.get_bounded_early_index_params(
        early_index_u_params
    )
    late_index_params = late_index_pop.get_bounded_late_index_params(
        late_index_u_params
    )
    cov_params = covariance_kernels.get_bounded_cov_params(cov_u_params)

    component_model_params = (
        ftpt0_cens_params,
        tpc_params,
        logm0_params,
        logtc_params,
        early_index_params,
        late_index_params,
        cov_params,
    )
    diffmahpop_params = DEFAULT_DIFFMAHPOP_PARAMS._make(DEFAULT_DIFFMAHPOP_PARAMS)
    for params in component_model_params:
        diffmahpop_params = diffmahpop_params._replace(**params._asdict())

    return diffmahpop_params


@jjit
def get_diffmahpop_u_params_from_params(diffmahpop_params):
    component_model_params = get_component_model_params(diffmahpop_params)
    ftpt0_params, tpc_params, logm0_params = component_model_params[:3]
    logtc_params = component_model_params[3]
    early_index_params, late_index_params = component_model_params[4:6]
    cov_params = component_model_params[6]

    ftpt0_u_params = ftpt0_cens.get_unbounded_ftpt0_params(ftpt0_params)
    tpc_u_params = tp_pdf_cens.get_unbounded_tp_cens_params(tpc_params)
    logm0_u_params = logm0_pop.get_unbounded_m0pop_params(logm0_params)
    logtc_u_params = logtc_pop.get_unbounded_logtc_params(logtc_params)
    early_index_u_params = early_index_pop.get_unbounded_early_index_params(
        early_index_params
    )
    late_index_u_params = late_index_pop.get_unbounded_late_index_params(
        late_index_params
    )
    cov_u_params = covariance_kernels.get_unbounded_cov_params(cov_params)

    component_model_u_params = (
        ftpt0_u_params,
        tpc_u_params,
        logm0_u_params,
        logtc_u_params,
        early_index_u_params,
        late_index_u_params,
        cov_u_params,
    )
    diffmahpop_u_params = DEFAULT_DIFFMAHPOP_U_PARAMS._make(DEFAULT_DIFFMAHPOP_U_PARAMS)
    for u_params in component_model_u_params:
        diffmahpop_u_params = diffmahpop_u_params._replace(**u_params._asdict())

    return diffmahpop_u_params


@jjit
def _get_all_diffmahpop_params_from_varied(
    varied_params, default_params=DEFAULT_DIFFMAHPOP_PARAMS
):
    diffmahpop_params = default_params._replace(**varied_params._asdict())
    return diffmahpop_params


def get_varied_params_by_exclusion(all_params, excluded_pnames):
    gen = zip(all_params._fields, all_params)
    varied_pdict = OrderedDict(
        [(name, float(x)) for (name, x) in gen if name not in excluded_pnames]
    )
    VariedParams = namedtuple("VariedParams", varied_pdict.keys())
    varied_params = VariedParams(**varied_pdict)
    return varied_params
