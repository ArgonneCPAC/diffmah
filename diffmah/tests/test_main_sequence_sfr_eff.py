"""
"""
import functools
import numpy as np
from copy import deepcopy
from jax import value_and_grad
from jax import numpy as jax_np
from jax import jit as jax_jit
from ..main_sequence_sfr_eff import _log_sfr_efficiency_ms_jax
from ..main_sequence_sfr_eff import DEFAULT_SFR_MS_PARAMS, MEAN_SFR_MS_PARAMS
from ..main_sequence_sfr_eff import mean_log_sfr_efficiency_main_sequence
from ..main_sequence_sfr_eff import log_sfr_efficiency_main_sequence
from ..main_sequence_sfr_eff import mean_log_sfr_efficiency_ms_jax


def test_log_sfr_eff_ms():
    logt = np.linspace(-1, 1.141, 200)
    params = np.array(list(DEFAULT_SFR_MS_PARAMS.values())).astype("f4")
    log_sfr_eff = _log_sfr_efficiency_ms_jax(logt, *params)
    assert log_sfr_eff.size == logt.size
    assert np.all(np.isfinite(log_sfr_eff))


def test_log_sfr_eff_ms_wrapper():
    logt = np.linspace(-1, 1.141, 200)
    log_sfr_eff = log_sfr_efficiency_main_sequence(logt)
    assert log_sfr_eff.size == logt.size
    assert np.all(np.isfinite(log_sfr_eff))


def test_mean_log_sfr_eff_ms_wrapper():
    logmp = 12
    logt = np.linspace(-1, 1.14, 50)
    mean_log_sfr_eff = mean_log_sfr_efficiency_main_sequence(logt, logmp)
    assert np.all(np.isfinite(mean_log_sfr_eff))


def test_sfr_efficiency_responds_to_params():
    logt = np.linspace(-1, 1.14, 20)

    log_sfr_eff = log_sfr_efficiency_main_sequence(logt, lge0=-1)

    log_sfr_eff_lge0 = log_sfr_efficiency_main_sequence(logt, lge0=-2)
    assert not np.allclose(log_sfr_eff, log_sfr_eff_lge0)

    log_sfr_eff_k_early = log_sfr_efficiency_main_sequence(logt, k_early=3)
    assert not np.allclose(log_sfr_eff, log_sfr_eff_k_early)

    log_sfr_eff_lgtc = log_sfr_efficiency_main_sequence(logt, lgtc=1)
    assert not np.allclose(log_sfr_eff, log_sfr_eff_lgtc)

    log_sfr_eff_k_trans = log_sfr_efficiency_main_sequence(logt, k_trans=5)
    assert not np.allclose(log_sfr_eff, log_sfr_eff_k_trans)

    log_sfr_eff_a_late = log_sfr_efficiency_main_sequence(logt, a_late=-2)
    assert not np.allclose(log_sfr_eff, log_sfr_eff_a_late)


def test_mean_sfr_efficiency_responds_to_params():
    param_dict_fid = deepcopy(MEAN_SFR_MS_PARAMS)
    logmparr = (10, 12, 15)
    logt = np.linspace(-1, 1.14, 10)

    perfect_match = True
    for key, default in MEAN_SFR_MS_PARAMS.items():
        perfect_match = False
        for logmp in logmparr:
            d = dict()
            d[key] = param_dict_fid[key] / 2 - 0.1
            y1 = mean_log_sfr_efficiency_main_sequence(logt, logmp)
            y2 = mean_log_sfr_efficiency_main_sequence(logt, logmp, **d)
            if not np.allclose(y1, y2):
                perfect_match = False
        assert not perfect_match, key


def test_sfr_efficiency_is_differentiable():
    """
    """

    @functools.partial(jax_jit, static_argnums=(1,))
    def mse_loss(params, data):
        logt, target = data
        lge0, k_early, lgtc, lgec, k_trans, a_late = params
        log_sfr_eff = _log_sfr_efficiency_ms_jax(
            logt, lge0, k_early, lgtc, lgec, k_trans, a_late
        )
        diff = target - log_sfr_eff
        return jax_np.sum(diff) / diff.size

    npts = 250
    logt0 = np.log10(13.85)
    logt = np.linspace(-1, logt0, npts)
    params = np.array(list(DEFAULT_SFR_MS_PARAMS.values())).astype("f4")
    target = _log_sfr_efficiency_ms_jax(logt, *params)
    data = logt, target
    loss_init, grads = value_and_grad(mse_loss, argnums=0)(params, data)
    params_new = params - 0.05 * grads
    loss_new, grads = value_and_grad(mse_loss, argnums=0)(params_new, data)
    assert loss_new < loss_init


def test_mean_sfr_efficiency_is_differentiable():
    @functools.partial(jax_jit, static_argnums=(1,))
    def mse_loss(params, data):
        logmp, logt, target = data
        log_sfr_eff = mean_log_sfr_efficiency_ms_jax(logt, logmp, *params)
        diff = target - log_sfr_eff
        return jax_np.sum(diff) / diff.size

    npts = 250
    logmp = 12
    logt0 = np.log10(13.85)
    logt = np.linspace(-1, logt0, npts)
    params = np.array(list(MEAN_SFR_MS_PARAMS.values())).astype("f4")
    target = mean_log_sfr_efficiency_ms_jax(logt, logmp, *params)
    data = logmp, logt, target
    loss_init, grads = value_and_grad(mse_loss, argnums=0)(params, data)
    params_new = params - 0.05 * grads
    loss_new, grads = value_and_grad(mse_loss, argnums=0)(params_new, data)
    assert loss_new < loss_init
