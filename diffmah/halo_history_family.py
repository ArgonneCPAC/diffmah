"""
"""
import numpy as np
from jax import jit as jjit
from jax import lax, vmap

from diffmah.halo_assembly import (
    DEFAULT_MAH_PARAMS,
    _get_dt_array,
    _individual_halo_assembly_jax_kern,
)


@jjit
def _get_dmhdt_indices(logmp, coeff_eval0, coeff_eval1):
    """ """
    early_indx_mu_diff = (29 / 25) * coeff_eval0 - (1 / 3) * coeff_eval1
    late_indx_mu_diff = (21 / 20) * coeff_eval0 - (3 / 4) * coeff_eval1

    mu_early_index = _sigmoid(logmp, 12.6, 0.8, -0.5, 2)
    mu_late_index = _sigmoid(logmp, 13.5, 2, -1.25, 0.5)

    early_indx = mu_early_index + early_indx_mu_diff
    late_indx = mu_late_index + late_indx_mu_diff
    return early_indx, late_indx


@jjit
def _generate_halo_history_family_indx(
    logt, dtarr, logmp, dmhdt_x0, dmhdt_k, assembly_param, indx_tmp
):
    dmhdt_early_index, dmhdt_late_index = _get_dmhdt_indices(logmp, assembly_param, 0)
    return _individual_halo_assembly_jax_kern(
        logt,
        dtarr,
        logmp,
        dmhdt_x0,
        dmhdt_k,
        dmhdt_early_index,
        dmhdt_late_index,
        indx_tmp,
    )


_halo_assembly_mass_family_kern = vmap(
    _generate_halo_history_family_indx,
    in_axes=(None, None, 0, None, None, None, None),
)

_halo_assembly_index_family_kern = vmap(
    _halo_assembly_mass_family_kern,
    in_axes=(None, None, None, None, None, 0, None),
)

_halo_assembly_tmp_family_kern = vmap(
    _halo_assembly_index_family_kern,
    in_axes=(None, None, None, None, None, None, 0),
)


@jjit
def _generate_halo_history_family(
    logmp_family,
    assembly_family,
    indx_tmp_family,
    logt,
    dtarr,
    dmhdt_x0,
    dmhdt_k,
):
    return _halo_assembly_tmp_family_kern(
        logt,
        dtarr,
        logmp_family,
        dmhdt_x0,
        dmhdt_k,
        assembly_family,
        indx_tmp_family,
    )


def generate_halo_history_family(
    logmp_family,
    assembly_family,
    tmp_family,
    cosmic_time,
    dmhdt_x0=DEFAULT_MAH_PARAMS["dmhdt_x0"],
    dmhdt_k=DEFAULT_MAH_PARAMS["dmhdt_k"],
):
    """Generate a family of halo histories

    Parameters
    ----------
    logmp_family : ndarray, shape (n_mpeak, )

    assembly_family : ndarray, shape (n_assem, )

    tmp_family : ndarray, shape (n_tmp, )

    cosmic_time : ndarray, shape (n_times, )

    Returns
    -------
    log_mah : ndarray, shape (n_tmp, n_assem, n_mpeak, n_times)
        Base-10 log of halo mass in units of Msun

    log_dmhdt : ndarray, shape (n_tmp, n_assem, n_mpeak, n_times)
        Base-10 log of halo mass accretion rate in units of Msun/yr

    """
    indx_tmp_family = np.array([np.argmin(np.abs(t - cosmic_time)) for t in tmp_family])
    logt = np.log10(cosmic_time)
    dtarr = _get_dt_array(cosmic_time)
    return _generate_halo_history_family(
        logmp_family,
        assembly_family,
        indx_tmp_family,
        logt,
        dtarr,
        dmhdt_x0,
        dmhdt_k,
    )


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + lax.exp(-k * (x - x0)))
