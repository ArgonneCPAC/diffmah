"""
"""
# flake8: noqa
from collections import OrderedDict

import numpy as np

TODAY = 13.8
LGT0 = np.log10(TODAY)


DEFAULT_MAH_PDICT = OrderedDict(logmp=12.0, logtc=0.05, early_index=2.5, late_index=1.0)
DEFAULT_MAH_PARAMS = np.array(list(DEFAULT_MAH_PDICT.values()))
