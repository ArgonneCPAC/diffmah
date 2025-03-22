""" """

import numpy as np

from ...defaults import DEFAULT_MAH_PARAMS
from .. import param_utils as pu


def test_mc_select_diffmah_params():
    mc_is_1 = np.concatenate((np.ones(5), np.zeros(5)))
    ngals = mc_is_1.size
    ZZ1 = np.zeros(ngals)
    ZZ2 = np.zeros(ngals)

    mah_params_1 = DEFAULT_MAH_PARAMS._make(
        [getattr(DEFAULT_MAH_PARAMS, x) + ZZ1 for x in DEFAULT_MAH_PARAMS._fields]
    )
    mah_params_0 = DEFAULT_MAH_PARAMS._make(
        [getattr(DEFAULT_MAH_PARAMS, x) + ZZ2 for x in DEFAULT_MAH_PARAMS._fields]
    )
    mah_params = pu.mc_select_diffmah_params(mah_params_1, mah_params_0, mc_is_1)

    for pname in mah_params._fields:
        val = getattr(mah_params, pname)
        val1 = getattr(mah_params_1, pname)
        val0 = getattr(mah_params_0, pname)
        assert np.allclose(val[:5], val1[:5])
        assert np.allclose(val[5:], val0[5:])
