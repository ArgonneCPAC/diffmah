"""
"""
import pytest
import numpy as np
from jax import random as jran
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad
from ..monte_carlo_halo_population import mc_halo_population
from ..monte_carlo_halo_population import _mc_early_type_halo_mahs
from ..monte_carlo_halo_population import _mc_late_type_halo_mahs, _mc_halo_mahs
from ..rockstar_pdf_model import DEFAULT_MAH_PDF_PARAMS


SEED = 43


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_mc_halopop_is_differentiable():
    n_halos, n_times = 500, 50
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]

    lgm0 = np.zeros(n_halos) + 12.0
    mc_halopop_target = mc_halo_population(
        tarr, t0, lgm0, mah_type="early", seed=SEED + 1
    )
    mean_log_mah_target = np.mean(mc_halopop_target.log_mah, axis=0)

    @jjit
    def _mse(pred, target):
        diff = pred - target
        return jnp.mean(diff**2)

    @jjit
    def _loss(params, data):
        key, t, lgm, lgt0, log_mah_target = data
        mc_halopop = _mc_late_type_halo_mahs(key, t, lgm, lgt0, params)
        mse = _mse(mc_halopop.log_mah, log_mah_target)
        return mse

    _loss_and_grad = value_and_grad(_loss, argnums=0)

    mah_pdf_pdict = DEFAULT_MAH_PDF_PARAMS.copy()
    mah_pdf_pdict["frac_late_ylo"] = 0.25
    mah_pdf_pdict["frac_late_yhi"] = 0.85
    mah_pdf_params = np.array(list(mah_pdf_pdict.values()))

    loss_key = jran.PRNGKey(SEED)
    loss_data = loss_key, tarr, lgm0, np.log10(t0), mean_log_mah_target
    loss, grads = _loss_and_grad(mah_pdf_params, loss_data)
    assert loss > 0
    assert np.all(np.isfinite(grads))
    assert not np.all(grads == 0)


def test_diff_nondiff_mc_halopop_agrees():
    n_halos, n_times = 5, 6
    tarr = np.linspace(1, 13.8, n_times)
    t0 = tarr[-1]
    lgt0 = np.log10(t0)

    for lgm in (11, 15):
        lgm0 = lgm + np.zeros(n_halos)

        mah_pdf_pdict = DEFAULT_MAH_PDF_PARAMS.copy()
        mah_pdf_pdict["frac_late_ylo"] = 0.25
        mah_pdf_pdict["frac_late_yhi"] = 0.85
        mah_pdf_params = np.array(list(mah_pdf_pdict.values()))

        mc_halopop = mc_halo_population(tarr, t0, lgm0, seed=SEED, **mah_pdf_pdict)
        assert np.allclose(mc_halopop.log_mah[:, -1], lgm)

        ran_key = jran.PRNGKey(SEED)
        log_mah2 = _mc_halo_mahs(ran_key, tarr, lgm0, lgt0, mah_pdf_params)

        assert np.allclose(mc_halopop.log_mah[0, :], log_mah2[0, :], rtol=1e-4)
