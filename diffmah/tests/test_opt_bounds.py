"""
"""
from ..optimize_nbody import BOUNDS as NBODY_BOUNDS
from ..optimize_tng import BOUNDS as TNG_BOUNDS
from ..tng_pdf_model import DEFAULT_MAH_PDF_PARAMS as TNG_DEFAULTS
from ..rockstar_pdf_model import DEFAULT_MAH_PDF_PARAMS as NBODY_DEFAULTS


def test_nbody_params_are_correctly_bounded():
    for key, bounds in NBODY_BOUNDS.items():
        assert bounds[0] <= NBODY_DEFAULTS[key] <= bounds[1]


def test_tng_params_are_correctly_bounded():
    for key, bounds in TNG_BOUNDS.items():
        assert bounds[0] <= TNG_DEFAULTS[key] <= bounds[1]
