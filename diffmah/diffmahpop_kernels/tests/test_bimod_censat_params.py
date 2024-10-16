"""
"""

import numpy as np

from .. import bimod_censat_params as dpp

TOL = 1e-4


def test_diffmahpop_default_params_u_params_consistency():
    model_params = dpp.get_component_model_params(dpp.DEFAULT_DIFFMAHPOP_PARAMS)
    assert len(model_params) == len(dpp.COMPONENT_U_PDICTS)

    # enforce no name collisions on model params
    pdict_sizes = [len(x) for x in dpp.COMPONENT_PDICTS]
    assert sum(pdict_sizes) == len(dpp.DEFAULT_DIFFMAHPOP_PARAMS)

    u_params = dpp.get_diffmahpop_u_params_from_params(dpp.DEFAULT_DIFFMAHPOP_PARAMS)
    assert np.allclose(u_params, dpp.DEFAULT_DIFFMAHPOP_U_PARAMS, rtol=TOL)

    params = dpp.get_diffmahpop_params_from_u_params(dpp.DEFAULT_DIFFMAHPOP_U_PARAMS)
    assert np.allclose(params, dpp.DEFAULT_DIFFMAHPOP_PARAMS, rtol=TOL)
