"""
"""
# flake8: noqa
from collections import OrderedDict, namedtuple

import numpy as np

TODAY = 13.8
LGT0 = np.log10(TODAY)


DEFAULT_MAH_PDICT = OrderedDict(
    logmp=12.0, logtc=0.05, early_index=2.6137643, late_index=0.12692805
)
DiffmahParams = namedtuple("DiffmahParams", list(DEFAULT_MAH_PDICT.keys()))
DEFAULT_MAH_PARAMS = DiffmahParams(*list(DEFAULT_MAH_PDICT.values()))

MAH_K = 3.5
