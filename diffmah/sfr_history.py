"""
"""
from jax import numpy as jax_np
from .halo_assembly import _mean_halo_assembly_function
from .halo_assembly import MEAN_MAH_PARAMS, _get_individual_mah_params
from .sfr_efficiency import mean_log_sfr_efficiency_ms_jax, MEAN_SFR_MS_PARAMS
from .quenching_history import _mean_log_main_sequence_fraction, MEAN_Q_PARAMS
from .utils import _get_param_dict

FB = 0.158


def get_galaxy_history(logm0, cosmic_time, **kwargs):
    """Star formation rate and stellar mass as a function of time
    for a central galaxy living in a halo with present-day mass logm0.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0 in units of Msun.

    cosmic_time : ndarray of shape (n, )
        Age of the universe in Gyr at which to evaluate the assembly history.

    Returns
    -------
    log_sfr : ndarray of shape (n, )
        Base-10 log of SFR in units of Msun/yr

    log_sm : ndarray of shape (n, )
        Base-10 log of in-situ stellar mass in units of Msun

    """
    raise NotImplementedError()
