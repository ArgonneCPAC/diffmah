"""Sigmoid-based models for quenching probabilities."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
import jax

from .utils import get_1d_arrays, jax_sigmoid


__all__ = ("quenching_prob", "quenching_prob_cens", "quenching_prob_sats")


DEFAULT_PARAMS = OrderedDict(
    fq_cens_logm_crit=12.65,
    fq_cens_k=10 ** 0.2,
    fq_cens_ylo=0.15,
    fq_cens_yhi=0.9,
    fq_satboost_logmhost_crit=13,
    fq_satboost_logmhost_k=10 ** 0.2,
    fq_satboost_clusters=0.3,
    fq_sat_delay_time=2,
    fq_sat_tinfall_k=10 ** 0.5,
)

PARAM_BOUNDS = OrderedDict(
    fq_cens_logm_crit=(11, 15),
    fq_cens_k=(0, 2),
    fq_cens_ylo=(0, 1),
    fq_cens_yhi=(0, 1),
    fq_satboost_logmhost_crit=(11, 15),
    fq_satboost_logmhost_k=(0, 2),
    fq_satboost_clusters=(0, 1),
    fq_sat_delay_time=(0, 10),
    fq_sat_tinfall_k=(0, 2),
)


DEFAULT_CENS_PARAMS = OrderedDict(
    [(key, DEFAULT_PARAMS[key]) for key in DEFAULT_PARAMS.keys() if "sat" not in key]
)


def quenching_prob(
    upid,
    logmpeak,
    logmhost,
    time_since_infall,
    fq_cens_logm_crit=None,
    fq_cens_k=None,
    fq_cens_ylo=None,
    fq_cens_yhi=None,
    fq_satboost_logmhost_crit=None,
    fq_satboost_logmhost_k=None,
    fq_satboost_clusters=None,
    fq_sat_delay_time=None,
    fq_sat_tinfall_k=None,
):
    """Probability that a galaxy is quenched.

    Parameters
    ----------
    upid : float or ndarray of shape (nhalos, )
        The ID of the parent halo of a (sub)halo. Should be -1 to indicate a
        (sub)halo has no parents.
    logmpeak : ndarray, shape (nhalos, )
        Base-10 log of peak (sub)halo mass
    logmhost : ndarray, shape (nhalos, )
        Base-10 log of host halo mass
    time_since_infall : float or ndarray of shape (nhalos, )
        Time since infall for satellites.
    fq_cens_logm_crit : float, optional
        Value of log10(Mhalo) of the inflection point of qprob
    fq_cens_k : float, optional
        steepness of the sigmoid
    fq_cens_ylo : float, optional
        Quenched fraction of dwarf-mass centrals
    fq_cens_yhi : float, optional
        Quenched fraction of cluster-mass centrals
    fq_satboost_logmhost_crit : float, optional
        Value of log10(mhost) of the inflection point
        of the satellite quenching boost
    fq_satboost_logmhost_k : float, optional
        steepness of the sigmoid controlling the satellite quenching boost
    fq_satboost_clusters : float, optional
        Boost to quenched fraction of satellites of cluster-mass halos
    fq_sat_delay_time : float, optional
        Time after infall (in Gyr) when satellite-specific quenching begins
    fq_sat_tinfall_k : float, optional
        Steepness of the sigmoid function controlling how rapidly
        satellite-specific transition to becoming quenched

    Returns
    -------
    qprob : ndarray, shape (nhalos, )
        Probability of an object being quenched.
    """
    upid, logmpeak, logmhost, time_since_infall = get_1d_arrays(
        upid, logmpeak, logmhost, time_since_infall
    )

    fq_cens_logm_crit = (
        DEFAULT_PARAMS["fq_cens_logm_crit"]
        if fq_cens_logm_crit is None
        else fq_cens_logm_crit
    )
    fq_cens_k = DEFAULT_PARAMS["fq_cens_k"] if fq_cens_k is None else fq_cens_k
    fq_cens_ylo = DEFAULT_PARAMS["fq_cens_ylo"] if fq_cens_ylo is None else fq_cens_ylo
    fq_cens_yhi = DEFAULT_PARAMS["fq_cens_yhi"] if fq_cens_yhi is None else fq_cens_yhi

    fq_satboost_logmhost_crit = (
        DEFAULT_PARAMS["fq_satboost_logmhost_crit"]
        if fq_satboost_logmhost_crit is None
        else fq_satboost_logmhost_crit
    )
    fq_satboost_logmhost_k = (
        DEFAULT_PARAMS["fq_satboost_logmhost_k"]
        if fq_satboost_logmhost_k is None
        else fq_satboost_logmhost_k
    )
    fq_satboost_clusters = (
        DEFAULT_PARAMS["fq_satboost_clusters"]
        if fq_satboost_clusters is None
        else fq_satboost_clusters
    )
    fq_sat_delay_time = (
        DEFAULT_PARAMS["fq_sat_delay_time"]
        if fq_sat_delay_time is None
        else fq_sat_delay_time
    )
    fq_sat_tinfall_k = (
        DEFAULT_PARAMS["fq_sat_tinfall_k"]
        if fq_sat_tinfall_k is None
        else fq_sat_tinfall_k
    )

    params = np.array(
        [
            fq_cens_logm_crit,
            fq_cens_k,
            fq_cens_ylo,
            fq_cens_yhi,
            fq_satboost_logmhost_crit,
            fq_satboost_logmhost_k,
            fq_satboost_clusters,
            fq_sat_delay_time,
            fq_sat_tinfall_k,
        ]
    )
    return np.asarray(
        quenching_prob_jax(upid, logmpeak, logmhost, time_since_infall, params)
    )


def _quenching_prob_jax_kern(upid, logmpeak, logmhost, time_since_infall, params):
    """Probability that a galaxy is quenched.

    Parameters
    ----------
    upid : float or ndarray of shape (nhalos, )
        The ID of the parent halo of a (sub)halo. Should be -1 to indicate a
        (sub)halo has no parents.
    logmpeak : ndarray, shape (nhalos, )
        Base-10 log of peak (sub)halo mass
    logmhost : ndarray, shape (nhalos, )
        Base-10 log of host halo mass
    time_since_infall : float or ndarray of shape (nhalos, )
        Time since infall for satellites.
    params : array, shape (9,)
        An array with the parameters

            fq_cens_logm_crit
            fq_cens_k
            fq_cens_ylo
            fq_cens_yhi
            fq_satboost_logmhost_crit
            fq_satboost_logmhost_k
            fq_satboost_clusters
            fq_sat_delay_time
            fq_sat_tinfall_k

        See the documentation of the function `quenching_prob` for
        their definitions.

    Returns
    -------
    qprob : ndarray, shape (nhalos, )
        Probability of an object being quenched.
    """
    return jax_np.where(
        upid == -1,
        # centrals
        _quenching_prob_cens_jax_kern(logmpeak, params),
        # sats
        _quenching_prob_sats_jax_kern(logmpeak, logmhost, time_since_infall, params),
    )


quenching_prob_jax = jax.jit(
    jax.vmap(_quenching_prob_jax_kern, in_axes=(0, 0, 0, 0, None))
)
quenching_prob_jax.__doc__ = _quenching_prob_jax_kern.__doc__


def quenching_prob_cens(
    logmhalo,
    fq_cens_logm_crit=None,
    fq_cens_k=None,
    fq_cens_ylo=None,
    fq_cens_yhi=None,
):
    """Probability that a central galaxy is quenched.

    Parameters
    ----------
    logmhalo : ndarray, shape (nhalos, )
        Base-10 log of host halo mass
    fq_cens_logm_crit : float, optional
        Value of log10(Mhalo) of the inflection point of qprob
    fq_cens_k : float, optional
        steepness of the sigmoid
    fq_cens_ylo : float, optional
        Quenched fraction of dwarf-mass centrals
    fq_cens_yhi : float, optional
        Quenched fraction of cluster-mass centrals

    Returns
    -------
    qprob : ndarray
        Numpy array of shape (nhalos, )

    """
    (logmhalo,) = get_1d_arrays(logmhalo)

    fq_cens_logm_crit = (
        DEFAULT_PARAMS["fq_cens_logm_crit"]
        if fq_cens_logm_crit is None
        else fq_cens_logm_crit
    )
    fq_cens_k = DEFAULT_PARAMS["fq_cens_k"] if fq_cens_k is None else fq_cens_k
    fq_cens_ylo = DEFAULT_PARAMS["fq_cens_ylo"] if fq_cens_ylo is None else fq_cens_ylo
    fq_cens_yhi = DEFAULT_PARAMS["fq_cens_yhi"] if fq_cens_yhi is None else fq_cens_yhi

    params = np.array([fq_cens_logm_crit, fq_cens_k, fq_cens_ylo, fq_cens_yhi])

    return np.asarray(quenching_prob_cens_jax(logmhalo, params))


def _quenching_prob_cens_jax_kern(logmhalo, params):
    """Quenching probability for centrals.

    Parameters
    ----------
    logmhalo : ndarray of shape (nhalos,)
        Base-10 log of host halo mass
    params : array-like, shape (4,)
        An array with the parameters

            fq_cens_logm_crit
            fq_cens_k
            fq_cens_ylo
            fq_cens_yhi

        See the documentation of the function `quenching_prob` for
        their definitions.

    Returns
    -------
    qprob : ndarray of shape (nhalos,)
        The probability that a given (sub)halo has been quenched.
    """
    x0 = params[0]  # fq_cens_logm_crit
    k = params[1]  # fq_cens_k
    ylo = params[2]  # fq_cens_ylo
    yhi = params[3]  # fq_cens_yhi
    return jax_sigmoid(logmhalo, x0, k, ylo, yhi)


quenching_prob_cens_jax = jax.jit(
    jax.vmap(_quenching_prob_cens_jax_kern, in_axes=(0, None))
)
quenching_prob_cens_jax.__doc__ = _quenching_prob_cens_jax_kern.__doc__


def quenching_prob_sats(
    logmpeak,
    logmhost,
    time_since_infall,
    fq_cens_logm_crit=None,
    fq_cens_k=None,
    fq_cens_ylo=None,
    fq_cens_yhi=None,
    fq_satboost_logmhost_crit=None,
    fq_satboost_logmhost_k=None,
    fq_satboost_clusters=None,
    fq_sat_delay_time=None,
    fq_sat_tinfall_k=None,
):
    """Probability that a satellite galaxy is quenched.

    Parameters
    ----------
    logmpeak : ndarray, shape (nhalos, )
        Base-10 log of peak (sub)halo mass
    logmhost : ndarray, shape (nhalos, )
        Base-10 log of host halo mass
    time_since_infall : float or ndarray of shape (nhalos, )
        Time since infall for satellites.
    fq_cens_logm_crit : float, optional
        Value of log10(Mhalo) of the inflection point of qprob
    fq_cens_k : float, optional
        steepness of the sigmoid
    fq_cens_ylo : float, optional
        Quenched fraction of dwarf-mass centrals
    fq_cens_yhi : float, optional
        Quenched fraction of cluster-mass centrals
    fq_satboost_logmhost_crit : float, optional
        Value of log10(mhost) of the inflection point
        of the satellite quenching boost
    fq_satboost_logmhost_k : float, optional
        steepness of the sigmoid controlling the satellite quenching boost
    fq_satboost_clusters : float, optional
        Boost to quenched fraction of satellites of cluster-mass halos
    fq_sat_delay_time : float, optional
        Time after infall (in Gyr) when satellite-specific quenching begins
    fq_sat_tinfall_k : float, optional
        steepness of the sigmoid function controlling how rapidly
        satellite-specific transition to becoming quenched

    Returns
    -------
    qprob : ndarray, shape (nhalos, )
        Probability of an object being quenched.
    """
    logmpeak, logmhost, time_since_infall = get_1d_arrays(
        logmpeak, logmhost, time_since_infall
    )

    fq_cens_logm_crit = (
        DEFAULT_PARAMS["fq_cens_logm_crit"]
        if fq_cens_logm_crit is None
        else fq_cens_logm_crit
    )
    fq_cens_k = DEFAULT_PARAMS["fq_cens_k"] if fq_cens_k is None else fq_cens_k
    fq_cens_ylo = DEFAULT_PARAMS["fq_cens_ylo"] if fq_cens_ylo is None else fq_cens_ylo
    fq_cens_yhi = DEFAULT_PARAMS["fq_cens_yhi"] if fq_cens_yhi is None else fq_cens_yhi

    fq_satboost_logmhost_crit = (
        DEFAULT_PARAMS["fq_satboost_logmhost_crit"]
        if fq_satboost_logmhost_crit is None
        else fq_satboost_logmhost_crit
    )
    fq_satboost_logmhost_k = (
        DEFAULT_PARAMS["fq_satboost_logmhost_k"]
        if fq_satboost_logmhost_k is None
        else fq_satboost_logmhost_k
    )
    fq_satboost_clusters = (
        DEFAULT_PARAMS["fq_satboost_clusters"]
        if fq_satboost_clusters is None
        else fq_satboost_clusters
    )
    fq_sat_delay_time = (
        DEFAULT_PARAMS["fq_sat_delay_time"]
        if fq_sat_delay_time is None
        else fq_sat_delay_time
    )
    fq_sat_tinfall_k = (
        DEFAULT_PARAMS["fq_sat_tinfall_k"]
        if fq_sat_tinfall_k is None
        else fq_sat_tinfall_k
    )

    params = np.array(
        [
            fq_cens_logm_crit,
            fq_cens_k,
            fq_cens_ylo,
            fq_cens_yhi,
            fq_satboost_logmhost_crit,
            fq_satboost_logmhost_k,
            fq_satboost_clusters,
            fq_sat_delay_time,
            fq_sat_tinfall_k,
        ]
    )
    return np.asarray(
        quenching_prob_sats_jax(logmpeak, logmhost, time_since_infall, params)
    )


def _quenching_prob_sats_jax_kern(logmpeak, logmhost, time_since_infall, params):
    """Probability that a satellite galaxy is quenched.

    Parameters
    ----------
    logmpeak : ndarray, shape (nhalos, )
        Base-10 log of peak (sub)halo mass
    logmhost : ndarray, shape (nhalos, )
        Base-10 log of host halo mass
    time_since_infall : float or ndarray of shape (nhalos, )
        Time since infall for satellites.
    params : array, shape (9,)
        An array with the parameters

            fq_cens_logm_crit
            fq_cens_k
            fq_cens_ylo
            fq_cens_yhi
            fq_satboost_logmhost_crit
            fq_satboost_logmhost_k
            fq_satboost_clusters
            fq_sat_delay_time
            fq_sat_tinfall_k

        See the documentation of the function `quenching_prob` for
        their definitions.

    Returns
    -------
    qprob : ndarray, shape (nhalos, )
        Probability of an object being quenched.
    """

    qprob_cens = _quenching_prob_cens_jax_kern(logmpeak, params[:4])

    qprob_boost_sats_limit = _quenching_prob_boost_sats_jax_kern(
        logmpeak, logmhost, params,
    )

    qprob_satboost_infall_factor = _qprob_sat_infall_dependence_jax_kern(
        time_since_infall, params,
    )

    qprob_boost_sats = qprob_boost_sats_limit * qprob_satboost_infall_factor

    return qprob_cens + (1 - qprob_cens) * qprob_boost_sats


quenching_prob_sats_jax = jax.jit(
    jax.vmap(_quenching_prob_sats_jax_kern, in_axes=(0, 0, 0, None))
)
quenching_prob_sats_jax.__doc__ = _quenching_prob_sats_jax_kern.__doc__


def _quenching_prob_boost_sats_jax_kern(logm, logmhost, params):
    x0, k = params[4], params[5]
    ylo, yhi = 0, params[6]
    return jax_sigmoid(logmhost, x0, k, ylo, yhi)


def _qprob_sat_infall_dependence_jax_kern(time_since_infall, params):
    x0, k = params[7], params[8]
    ylo, yhi = 0, 1
    return jax_sigmoid(time_since_infall, x0, k, ylo, yhi)
