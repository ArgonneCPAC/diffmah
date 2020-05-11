"""
"""
import functools
import numpy as np
from jax import numpy as jax_np
from jax import value_and_grad
from jax import jit as jax_jit
from ..halo_assembly import halo_mass_assembly_history, DEFAULT_MAH_PARAMS
from ..halo_assembly import mean_halo_mass_assembly_history, _halo_assembly_function


def test_halo_mah_evaluates_reasonably_with_default_args():
    """
    """
    npts = 250
    for logm0 in (11, 12, 13, 14, 15):
        for t0 in (13.5, 14):
            cosmic_time = np.linspace(0.1, t0, npts)
            logmah, log_dmhdt = halo_mass_assembly_history(logm0, cosmic_time, t0=t0)
            assert logmah.size == npts == log_dmhdt.size
            assert np.allclose(logmah[-1], logm0, atol=0.01)


def test_avg_halo_mah_evaluates_reasonably_with_default_args():
    """
    """
    npts = 250
    for logm0 in (11, 12, 13, 14, 15):
        for t0 in (13.5, 14):
            cosmic_time = np.linspace(0.1, t0, npts)
            logmah, log_dmhdt = mean_halo_mass_assembly_history(
                logm0, cosmic_time, t0=t0
            )
            assert logmah.size == npts == log_dmhdt.size
            assert np.allclose(logmah[-1], logm0, atol=0.01)


def test_halo_assembly_function_differentiability():
    """
    """

    @functools.partial(jax_jit, static_argnums=(1,))
    def mse_loss(mah_params, data):
        tarr, logm0, indx_t0, logt0, logmah_target = data
        logmah, log_dmhdt = _halo_assembly_function(
            mah_params, tarr, logm0, indx_t0, logt0
        )
        diff_logmah = logmah - logmah_target
        return jax_np.sum(diff_logmah * diff_logmah) / diff_logmah.size

    npts = 250
    logm0 = 12
    t0 = 13.85
    tarr = np.linspace(0.1, t0, npts)
    logt0 = np.log10(t0)
    indx_t0 = -1
    default_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    logmah_target = _halo_assembly_function(
        default_params, tarr, logm0, indx_t0, logt0
    )[0]

    data = tarr, logm0, indx_t0, logt0, logmah_target
    loss_fid = mse_loss(default_params, data)
    assert np.allclose(loss_fid, 0, atol=0.01)

    mah_params_init = np.array((0.2, 4, 0.5, -1.5))

    loss_init, mse_loss_grads = value_and_grad(mse_loss, argnums=0)(
        mah_params_init, data
    )
    assert loss_init > loss_fid
    new_params = mah_params_init - 0.05 * mse_loss_grads
    loss_new = mse_loss(new_params, data)
    assert loss_new < loss_init
