"""Module used to predict the stellar-to-halo-mass relation."""
from collections import OrderedDict
import numpy as np
from .in_situ_smh_kernel import in_situ_mstar_at_zobs
from .sigmoid_mah import LOGT0, DEFAULT_MAH_PARAMS
from .sigmoid_mah import _mah_sigmoid_params_logm_at_logt


def mstar_vs_mhalo_at_zobs(
    zobs, tobs, logmh_at_zobs, qtime=None, logt0=LOGT0, logtc=None, **kwargs,
):
    """Calculate in-situ M* for a halo with mass logmh_at_zobs at redshift zobs.

    Parameters
    ----------
    zobs : float

    tobs : float
        Age of the Universe in Gyr at zobs

    logmh_at_zobs : float
        Base-10 log of halo mass at zobs

    qtime : float, optional
        Quenching time in Gyr. Default is set by quenching time parameters.

    logt0 : float, optional
        Base-10 log of the age of the Universe at z=0.
        Default is 1.14 for Planck15 cosmology.
        Should be consistent with tobs.

    logtc : float, optional
        Base-10 log of the critical time in Gyr.
        Smaller values of logtc produce halos with earlier formation times.
        Default value is set by halo MAH parameters defined in the sigmoid_mah module.

    **model_params : float, optional
        Any parameter regulating main-sequence SFR or quenching is accepted.

    Returns
    -------
    mstar_at_zobs_ms : float
        M* at zobs for main-sequence galaxies

    mstar_at_zobs_median : float
        Weighted-average of main-sequence and quenched M*,
        weighting by the probability the galaxy is quenched.

    mstar_at_zobs_q : float
        M* at zobs for quenched galaxies

    """
    assert np.shape(zobs) == (), "zobs should be a scalar"
    assert np.shape(tobs) == (), "tobs should be a scalar"
    logtobs = np.log10(tobs)

    mah_params, sfr_params = _parse_args(**kwargs)

    logtc, logtk, dlogm_height, logm0 = _mah_sigmoid_params_logm_at_logt(
        logtobs, logmh_at_zobs, logtc=logtc, **mah_params
    )

    _arrs = _get_1d_arrays(logmh_at_zobs, logtc, logtk, dlogm_height, logm0)
    logmh_at_zobs, logtc, logtk, dlogm_height, logm0 = _arrs
    n = logm0.size

    if qtime is not None:
        qtime = qtime + np.zeros(n)

    mstar_at_zobs_quenched = np.zeros_like(logm0)
    mstar_at_zobs_median = np.zeros_like(logm0)
    mstar_at_zobs_ms = np.zeros_like(logm0)
    for i in range(n):
        if qtime is None:
            qt = qtime
        else:
            qt = qtime[i]

        mstar_ms, mstar_med, mstar_q = in_situ_mstar_at_zobs(
            zobs,
            logm0[i],
            qtime=qt,
            logt0=logt0,
            logtc=logtc[i],
            logtk=logtk[i],
            dlogm_height=dlogm_height[i],
            **sfr_params,
        )
        mstar_at_zobs_ms[i] = mstar_ms
        mstar_at_zobs_median[i] = mstar_med
        mstar_at_zobs_quenched[i] = mstar_q
        #

    return mstar_at_zobs_ms, mstar_at_zobs_median, mstar_at_zobs_quenched


def _parse_args(**kwargs):
    mah_params = OrderedDict()
    for key, default_val in DEFAULT_MAH_PARAMS.items():
        mah_params[key] = kwargs.get(key, default_val)

    sfr_params = OrderedDict(
        [(key, val) for key, val in kwargs.items() if key not in mah_params]
    )
    return mah_params, sfr_params


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length."""
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
