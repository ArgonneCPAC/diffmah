"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from . import (
    covariance_kernels,
    early_index_bimod,
    frac_early_cens,
    late_index_bimod,
    logtc_bimod,
)
from .bimod_logm0_kernels import logm0_pop_bimod
from .bimod_logm0_sats import logm0_pop_bimod_sats
from .t_peak_kernels import tp_pdf_cens_flex, tp_pdf_sats

DEFAULT_DIFFMAHPOP_PDICT = OrderedDict()
COMPONENT_PDICTS = (
    tp_pdf_cens_flex.DEFAULT_TPCENS_PDICT,
    tp_pdf_sats.DEFAULT_TP_SATS_PDICT,
    logm0_pop_bimod.DEFAULT_LOGM0_PDICT,
    logm0_pop_bimod_sats.DEFAULT_LOGM0_PDICT,
    logtc_bimod.LOGTC_PDICT,
    early_index_bimod.EARLY_INDEX_PDICT,
    late_index_bimod.LATE_INDEX_PDICT,
    frac_early_cens.DEFAULT_FEC_PDICT,
    covariance_kernels.DEFAULT_COV_PDICT,
)
for pdict in COMPONENT_PDICTS:
    DEFAULT_DIFFMAHPOP_PDICT.update(pdict)
DiffmahPop_Params = namedtuple("DiffmahPop_Params", DEFAULT_DIFFMAHPOP_PDICT.keys())
DEFAULT_DIFFMAHPOP_PARAMS = DiffmahPop_Params(**DEFAULT_DIFFMAHPOP_PDICT)


COMPONENT_U_PDICTS = (
    tp_pdf_cens_flex.DEFAULT_TPCENS_U_PARAMS._asdict(),
    tp_pdf_sats.DEFAULT_TP_SATS_U_PARAMS._asdict(),
    logm0_pop_bimod.DEFAULT_LOGM0POP_U_PARAMS._asdict(),
    logm0_pop_bimod_sats.DEFAULT_LOGM0POP_U_PARAMS._asdict(),
    logtc_bimod.DEFAULT_LOGTC_U_PARAMS._asdict(),
    early_index_bimod.DEFAULT_EARLY_INDEX_U_PARAMS._asdict(),
    late_index_bimod.DEFAULT_LATE_INDEX_U_PARAMS._asdict(),
    frac_early_cens.DEFAULT_FEC_U_PARAMS._asdict(),
    covariance_kernels.DEFAULT_COV_U_PARAMS._asdict(),
)
DEFAULT_DIFFMAHPOP_U_PDICT = OrderedDict()
for updict in COMPONENT_U_PDICTS:
    DEFAULT_DIFFMAHPOP_U_PDICT.update(updict)
DiffmahPop_UParams = namedtuple("DiffmahPop_UParams", DEFAULT_DIFFMAHPOP_U_PDICT.keys())
DEFAULT_DIFFMAHPOP_U_PARAMS = DiffmahPop_UParams(**DEFAULT_DIFFMAHPOP_U_PDICT)


@jjit
def get_component_model_params(diffmahpop_params):
    tp_pdf_cens_flex_params = tp_pdf_cens_flex.TPCens_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in tp_pdf_cens_flex.TPCens_Params._fields
        ]
    )
    tp_pdf_sats_params = tp_pdf_sats.TP_Sats_Params(
        *[getattr(diffmahpop_params, key) for key in tp_pdf_sats.TP_Sats_Params._fields]
    )
    logm0_params = logm0_pop_bimod.LGM0Pop_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in logm0_pop_bimod.LGM0Pop_Params._fields
        ]
    )

    logm0_params_sats = logm0_pop_bimod_sats.LGM0Pop_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in logm0_pop_bimod_sats.LGM0Pop_Params._fields
        ]
    )

    logtc_params = logtc_bimod.Logtc_Params(
        *[getattr(diffmahpop_params, key) for key in logtc_bimod.Logtc_Params._fields]
    )
    early_index_params = early_index_bimod.EarlyIndex_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in early_index_bimod.EarlyIndex_Params._fields
        ]
    )
    late_index_params = late_index_bimod.LateIndex_Params(
        *[
            getattr(diffmahpop_params, key)
            for key in late_index_bimod.LateIndex_Params._fields
        ]
    )

    fec_params = frac_early_cens.FEC_Params(
        *[getattr(diffmahpop_params, key) for key in frac_early_cens.FEC_Params._fields]
    )

    cov_params = covariance_kernels.CovParams(
        *[
            getattr(diffmahpop_params, key)
            for key in covariance_kernels.CovParams._fields
        ]
    )
    return (
        tp_pdf_cens_flex_params,
        tp_pdf_sats_params,
        logm0_params,
        logm0_params_sats,
        logtc_params,
        early_index_params,
        late_index_params,
        fec_params,
        cov_params,
    )


@jjit
def get_component_model_u_params(diffmahpop_u_params):
    tp_pdf_cens_flex_u_params = tp_pdf_cens_flex.TPCens_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in tp_pdf_cens_flex.TPCens_UParams._fields
        ]
    )
    tp_pdf_sats_u_params = tp_pdf_sats.TP_Sats_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in tp_pdf_sats.TP_Sats_UParams._fields
        ]
    )
    logm0_u_params = logm0_pop_bimod.LGM0Pop_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in logm0_pop_bimod.LGM0Pop_UParams._fields
        ]
    )

    logm0_sats_u_params = logm0_pop_bimod_sats.LGM0Pop_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in logm0_pop_bimod_sats.LGM0Pop_UParams._fields
        ]
    )

    logtc_u_params = logtc_bimod.Logtc_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in logtc_bimod.Logtc_UParams._fields
        ]
    )
    early_index_u_params = early_index_bimod.EarlyIndex_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in early_index_bimod.EarlyIndex_UParams._fields
        ]
    )
    late_index_u_params = late_index_bimod.LateIndex_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in late_index_bimod.LateIndex_UParams._fields
        ]
    )

    fec_u_params = frac_early_cens.FEC_UParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in frac_early_cens.FEC_UParams._fields
        ]
    )

    cov_u_params = covariance_kernels.CovUParams(
        *[
            getattr(diffmahpop_u_params, key)
            for key in covariance_kernels.CovUParams._fields
        ]
    )

    return (
        tp_pdf_cens_flex_u_params,
        tp_pdf_sats_u_params,
        logm0_u_params,
        logm0_sats_u_params,
        logtc_u_params,
        early_index_u_params,
        late_index_u_params,
        fec_u_params,
        cov_u_params,
    )


@jjit
def get_diffmahpop_params_from_u_params(diffmahpop_u_params):
    component_model_u_params = get_component_model_u_params(diffmahpop_u_params)
    tpc_u_params, tps_u_params, logm0_u_params, logm0_sats_u_params = (
        component_model_u_params[:4]
    )
    logtc_u_params = component_model_u_params[4]
    early_index_u_params, late_index_u_params = component_model_u_params[5:7]
    fec_u_params, cov_u_params = component_model_u_params[7:]

    tpc_params = tp_pdf_cens_flex.get_bounded_tp_cens_params(tpc_u_params)
    tps_params = tp_pdf_sats.get_bounded_tp_sat_params(tps_u_params)
    logm0_params = logm0_pop_bimod.get_bounded_m0pop_params(logm0_u_params)
    logm0_sats_params = logm0_pop_bimod_sats.get_bounded_m0pop_params(
        logm0_sats_u_params
    )
    logtc_params = logtc_bimod.get_bounded_logtc_params(logtc_u_params)
    early_index_params = early_index_bimod.get_bounded_early_index_params(
        early_index_u_params
    )
    late_index_params = late_index_bimod.get_bounded_late_index_params(
        late_index_u_params
    )
    fec_params = frac_early_cens.get_bounded_fec_params(fec_u_params)

    cov_params = covariance_kernels.get_bounded_cov_params(cov_u_params)

    component_model_params = (
        tpc_params,
        tps_params,
        logm0_params,
        logm0_sats_params,
        logtc_params,
        early_index_params,
        late_index_params,
        fec_params,
        cov_params,
    )
    diffmahpop_params = DEFAULT_DIFFMAHPOP_PARAMS._make(DEFAULT_DIFFMAHPOP_PARAMS)
    for params in component_model_params:
        diffmahpop_params = diffmahpop_params._replace(**params._asdict())

    return diffmahpop_params


@jjit
def get_diffmahpop_u_params_from_params(diffmahpop_params):
    component_model_params = get_component_model_params(diffmahpop_params)
    tpc_params, tps_params, logm0_params, logm0_sats_params = component_model_params[:4]
    logtc_params = component_model_params[4]
    early_index_params, late_index_params = component_model_params[5:7]
    fec_params, cov_params = component_model_params[7:]

    tpc_u_params = tp_pdf_cens_flex.get_unbounded_tp_cens_params(tpc_params)
    tps_u_params = tp_pdf_sats.get_unbounded_tp_sat_params(tps_params)
    logm0_u_params = logm0_pop_bimod.get_unbounded_m0pop_params(logm0_params)
    logm0_sats_u_params = logm0_pop_bimod_sats.get_unbounded_m0pop_params(
        logm0_sats_params
    )
    logtc_u_params = logtc_bimod.get_unbounded_logtc_params(logtc_params)
    early_index_u_params = early_index_bimod.get_unbounded_early_index_params(
        early_index_params
    )
    late_index_u_params = late_index_bimod.get_unbounded_late_index_params(
        late_index_params
    )
    fec_u_params = frac_early_cens.get_unbounded_fec_params(fec_params)
    cov_u_params = covariance_kernels.get_unbounded_cov_params(cov_params)

    component_model_u_params = (
        tpc_u_params,
        tps_u_params,
        logm0_u_params,
        logm0_sats_u_params,
        logtc_u_params,
        early_index_u_params,
        late_index_u_params,
        fec_u_params,
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
