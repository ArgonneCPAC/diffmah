"""
"""

import numpy as np


def test_default_mah_params_imports_from_top_level_and_is_frozen():
    from .. import DEFAULT_MAH_PARAMS

    assert np.allclose(DEFAULT_MAH_PARAMS, (12.0, 0.05, 2.6137643, 0.12692805))


def test_mah_k_imports_from_top_level():
    from .. import MAH_K

    assert np.allclose(MAH_K, 3.5)


def test_mah_halopop_imports_from_top_level():
    from .. import DEFAULT_MAH_PARAMS, DiffmahParams, mah_halopop

    tarr = np.linspace(0.1, 13.7, 100)
    ngals = 150
    zz = np.zeros(ngals)
    mah_params_halopop = DiffmahParams(*[x + zz for x in DEFAULT_MAH_PARAMS])
    dmhdt, log_mah = mah_halopop(mah_params_halopop, tarr)
    assert log_mah.shape == dmhdt.shape


def test_diffmah_fitter_imports():
    from .. import diffmah_fitter  # noqa
