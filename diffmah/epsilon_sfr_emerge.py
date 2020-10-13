"""
"""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from jax import jit as jjit
from jax import vmap as jvmap
from .utils import get_1d_arrays, _get_param_dict


DEFAULT_PARAMS = OrderedDict(
    sfr_eff_m_0=11.339,
    sfr_eff_m_z=0.692,
    sfr_eff_beta_0=3.344,
    sfr_eff_beta_z=-2.079,
    sfr_eff_gamma_0=0.966,
    sfr_eff_lgeps_n0=-2.3,
    sfr_eff_lgeps_nz=-0.16,
)


def sfr_efficiency_function(mhalo_at_z, z, **kwargs):
    """SFR efficiency function from Moster+17, arXiv:1705.05373.
    See Eqns 5-10.

    Parameters
    ----------
    mhalo_at_z : float or ndarray of shape (n, )
        Stores halo mass at the input redshift

    z : float or ndarray of shape (n, )

    params : model parameters, optional
        All keywords of DEFAULT_PARAMS defined at top of module are accepted

    Returns
    -------
    sfr_eff : ndarray of shape (n, )

    """
    mhalo_at_z, z = get_1d_arrays(mhalo_at_z, z)

    param_dict = _get_param_dict(DEFAULT_PARAMS, strict=True, **kwargs)
    params = np.array(list(param_dict.values()))

    data = mhalo_at_z, z
    return np.array(_sfr_efficiency_kernel(params, data))


@jjit
def _sfr_history_kernel(params, data):
    mhalo_at_z, z, lg_fb, lg_dmhdt = data
    sfr_eff_data = mhalo_at_z, z
    lg_sfr_eff = jax_np.log10(_sfr_efficiency_kernel(params, sfr_eff_data))
    return lg_sfr_eff + lg_fb + lg_dmhdt


@jjit
def _sfr_efficiency_kernel(params, data):
    mhalo_at_z, z = data
    m_0, m_z, beta_0, beta_z, gamma_0, lgeps_n0, lgeps_nz = params
    params_at_z = _get_params_at_z(
        z, m_0, m_z, beta_0, beta_z, gamma_0, lgeps_n0, lgeps_nz
    )
    sfr_eff = _sfr_efficiency_dbl_plaw(mhalo_at_z, *params_at_z)
    return sfr_eff


def _get_params_at_z(z, m_0, m_z, beta_0, beta_z, gamma_0, lgeps_n0, lgeps_nz):
    m_1_at_z = jax_np.power(10, _get_log_m1_param_at_z(z, m_0, m_z))
    beta_at_z = _get_beta_param_at_z(z, beta_0, beta_z)
    gamma_at_z = _get_gamma_param_at_z(z, gamma_0)
    eps_at_z = _get_eps_n_param_at_z(z, lgeps_n0, lgeps_nz)
    params_at_z = m_1_at_z, beta_at_z, gamma_at_z, eps_at_z
    return params_at_z


@jjit
def _sfr_efficiency_dbl_plaw(mhalo_at_z, m_1_at_z, beta_at_z, gamma_at_z, eps_at_z):
    x = mhalo_at_z / m_1_at_z
    numerator = 2 * eps_at_z
    denominator = jax_np.power(x, -beta_at_z) + jax_np.power(x, gamma_at_z)
    return numerator / denominator


@jjit
def _get_log_m1_param_at_z(z, m_0, m_z):
    return m_0 + m_z * z / (1 + z)


@jjit
def _get_beta_param_at_z(z, beta_0, beta_z):
    return beta_0 + beta_z * z / (1 + z)


@jjit
def _get_gamma_param_at_z(z, gamma_0):
    return gamma_0


@jjit
def _get_eps_n_param_at_z(z, lgeps_n0, lgeps_nz):
    eps_n0 = jax_np.power(10, lgeps_n0)
    eps_nz = jax_np.power(10, lgeps_nz)
    return eps_n0 + eps_nz * z / (1 + z)


@jjit
def _get_m_max(m_1_at_z, beta_at_z, gamma_at_z):
    return m_1_at_z * (beta_at_z / gamma_at_z) ** (1 / (beta_at_z + gamma_at_z))


_get_params_at_z_halopop = jvmap(
    jvmap(_get_params_at_z, in_axes=(0, None, None, None, None, None, None, None)),
    in_axes=(None, 0, 0, 0, 0, 0, 0, 0),
)
