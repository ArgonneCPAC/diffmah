""" """

from jax import numpy as jnp

from ..defaults import DEFAULT_MAH_PARAMS


def mc_select_diffmah_params(mah_params_1, mah_params_0, mc_is_1):
    """Select Monte Carlo realization of diffmah params

    Parameters
    ----------
    mah_params_1 : namedtuple of mah_params
        mah_params_1 stores diffmah params

    mah_params_0 : namedtuple of mah_params
        mah_params_0 stores diffmah params

    mc_is_1 : bool
        Boolean array, shape (n, )
        Equals 1 for mah_params1 and 0 for mah_params0

    Returns
    -------
    mah_params: namedtuple of mah_params

    """
    mah_params = [
        jnp.where(mc_is_1, getattr(mah_params_1, x), getattr(mah_params_0, x))
        for x in DEFAULT_MAH_PARAMS._fields
    ]
    mah_params = DEFAULT_MAH_PARAMS._make(mah_params)
    return mah_params
