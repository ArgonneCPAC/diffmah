"""
"""

# flake8: noqa

import numpy as np

from . import diffmah_kernels as dk

__all__ = ("LGT0", "DEFAULT_MAH_PDICT", "DiffmahParams", "DEFAULT_MAH_PARAMS", "MAH_K")

TODAY = 13.8
LGT0 = np.log10(TODAY)

DEFAULT_MAH_PDICT = dk.DEFAULT_MAH_PDICT
DiffmahParams = dk.DiffmahParams
DEFAULT_MAH_PARAMS = dk.DEFAULT_MAH_PARAMS
MAH_K = dk.MAH_K = 3.5
