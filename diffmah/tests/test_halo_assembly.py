"""
"""
import functools
import numpy as np
from jax import numpy as jax_np
from jax import value_and_grad
from jax import jit as jax_jit
from ..halo_assembly import individual_halo_assembly_history
from ..halo_assembly import mean_halo_mass_assembly_history
from ..halo_assembly import _individual_halo_assembly_jax_kern
from ..halo_assembly import _get_individual_mah_params, _get_dt_array
from ..halo_assembly import DEFAULT_MAH_PARAMS, MEAN_MAH_PARAMS
from ..utils import _get_param_array


def test_halo_mah_evaluates_reasonably_with_default_args():
    """
    """
    npts = 250
    for logm0 in (11, 12, 13, 14, 15):
        for t0 in (13.5, 14):
            cosmic_time = np.linspace(0.1, t0, npts)
            logmah, log_dmhdt = individual_halo_assembly_history(
                cosmic_time, logm0, t0=t0
            )
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
                cosmic_time, logm0, t0=t0
            )
            assert logmah.size == npts == log_dmhdt.size
            assert np.allclose(logmah[-1], logm0, atol=0.01)


def test_individual_halo_assembly_differentiability():
    """
    """

    @functools.partial(jax_jit, static_argnums=(1,))
    def mse_loss(mah_params, data):
        logm0, logt, dtarr, indx_t0, logt0, logmah_target = data
        dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index = mah_params
        logmah, log_dmhdt = _individual_halo_assembly_jax_kern(
            logm0,
            dmhdt_x0,
            dmhdt_k,
            dmhdt_early_index,
            dmhdt_late_index,
            logt,
            dtarr,
            indx_t0,
        )

        diff_logmah = logmah - logmah_target
        return jax_np.sum(diff_logmah * diff_logmah) / diff_logmah.size

    npts = 100
    logm0 = 12
    t0 = 13.85
    tarr = np.linspace(0.1, t0, npts)
    logt = np.log10(tarr)
    dtarr = _get_dt_array(tarr)
    logt0 = np.log10(t0)
    indx_t0 = -1
    default_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    dmhdt_x0, dmhdt_k, dmhdt_early_index, dmhdt_late_index = default_params
    logmah_target = _individual_halo_assembly_jax_kern(
        logm0,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        logt,
        dtarr,
        indx_t0,
    )[0]

    data = logm0, logt, dtarr, indx_t0, logt0, logmah_target
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


def test_mean_mah_params_match_default_params_at_logm0_12():
    """Enforce that the _get_individual_mah_params function
    returns DEFAULT_MAH_PARAMS when logM0 = 12, and not otherwise.
    """
    mean_mah_param_arr = _get_param_array(MEAN_MAH_PARAMS)
    default_mah_param_arr = _get_param_array(DEFAULT_MAH_PARAMS)
    individual_mah_params = _get_individual_mah_params(12, *mean_mah_param_arr)
    assert np.allclose(default_mah_param_arr, individual_mah_params, atol=0.001)
    individual_mah_params = _get_individual_mah_params(12.1, *mean_mah_param_arr)
    assert not np.allclose(default_mah_param_arr, individual_mah_params, atol=0.001)
