"""
"""
import numpy as np
from jax import random as jran
from ..monte_carlo_halo_population import mc_halo_population
from ..monte_carlo_halo_population import _mc_early_type_halo_mahs
from ..monte_carlo_halo_population import _mc_late_type_halo_mahs
from ..rockstar_pdf_model import DEFAULT_MAH_PDF_PARAMS


SEED = 43


def test_diff_nondiff_mc_halopop_early_agree():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    lgt0 = np.log10(t0)

    for lgm in (11, 12, 13, 14, 15):
        lgm0 = lgm + np.zeros(n_halos)

        mah_pdf_pdict = DEFAULT_MAH_PDF_PARAMS.copy()
        mah_pdf_pdict["frac_late_ylo"] = 0.25
        mah_pdf_pdict["frac_late_yhi"] = 0.85
        mah_pdf_params = np.array(list(mah_pdf_pdict.values()))

        mc_halopop = mc_halo_population(
            tarr, t0, lgm0, mah_type="early", seed=SEED, **mah_pdf_pdict
        )

        ran_key = jran.PRNGKey(SEED)
        early_key, late_key, frac_key, ran_key = jran.split(ran_key, 4)
        mc_halopop2 = _mc_early_type_halo_mahs(
            early_key, tarr, lgm0, lgt0, mah_pdf_params
        )

        assert np.allclose(mc_halopop.log_mah, mc_halopop2.log_mah, rtol=1e-4)
        assert np.allclose(mc_halopop.dmhdt, mc_halopop2.dmhdt, rtol=1e-4)


def test_diff_nondiff_mc_halopop_late_agree():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    lgt0 = np.log10(t0)

    for lgm in (11, 12, 13, 14, 15):
        lgm0 = lgm + np.zeros(n_halos)

        mah_pdf_pdict = DEFAULT_MAH_PDF_PARAMS.copy()
        mah_pdf_pdict["frac_late_ylo"] = 0.25
        mah_pdf_pdict["frac_late_yhi"] = 0.85
        mah_pdf_params = np.array(list(mah_pdf_pdict.values()))

        mc_halopop = mc_halo_population(
            tarr, t0, lgm0, mah_type="late", seed=SEED, **mah_pdf_pdict
        )

        ran_key = jran.PRNGKey(SEED)
        early_key, late_key, frac_key, ran_key = jran.split(ran_key, 4)
        mc_halopop2 = _mc_late_type_halo_mahs(
            late_key, tarr, lgm0, lgt0, mah_pdf_params
        )

        assert np.allclose(mc_halopop.log_mah, mc_halopop2.log_mah, rtol=1e-4)
        assert np.allclose(mc_halopop.dmhdt, mc_halopop2.dmhdt, rtol=1e-4)
