"""Model for the quenching time of central galaxies."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from jax.scipy.special import erfinv as jax_erfinv
from jax.scipy.special import erf as jax_erf
from jax import jit as jax_jit
from jax import vmap as jax_vmap
from .utils import _enforce_no_extraneous_keywords
from .utils import jax_inverse_sigmoid, _jax_triweight_sigmoid_kernel


DEFAULT_PARAMS = OrderedDict(
    qt_lgmc=12.25,
    qt_k=1.5,
    qt_dwarfs=15,
    qt_clusters=1,
    qt_scatter_lgmc=12.5,
    qt_scatter_k=1,
    qt_scatter_dwarfs=0.1,
    qt_scatter_clusters=0.4,
)
PARAM_BOUNDS = OrderedDict(
    qt_lgmc=(11.5, 13),
    qt_k=(0.5, 3),
    qt_dwarfs=(10, 20),
    qt_clusters=(0.1, 10),
    qt_scatter_lgmc=(11.5, 13.25),
    qt_scatter_k=(0.25, 4),
    qt_scatter_dwarfs=(0.01, 5),
    qt_scatter_clusters=(0.01, 1),
)

DEFAULT_CENS_PARAMS = OrderedDict(
    [(key, DEFAULT_PARAMS[key]) for key in DEFAULT_PARAMS.keys()]
)
PARAM_BOUNDS_CENS = OrderedDict(
    [(key, PARAM_BOUNDS[key]) for key in PARAM_BOUNDS.keys()]
)


QFUNC_PARAMS = OrderedDict(qfunc_k=5, qfunc_ylo=1, qfunc_yhi=0)


def quenching_function(
    t,
    qt,
    qfunc_k=QFUNC_PARAMS["qfunc_k"],
    qfunc_ylo=QFUNC_PARAMS["qfunc_ylo"],
    qfunc_yhi=QFUNC_PARAMS["qfunc_yhi"],
):
    """Sigmoid function smoothly dropping SFR from 1 to 0 at t = qt.

    Parameters
    ----------
    t : ndarray of shape (n, )
        Cosmic time

    qt : float or ndarray of shape (n, )
        Time of quenching

    Returns
    -------
    f : ndarray of shape (n, )
        Fraction of SFR remaining before/after the quenching event

    """
    t, qt = _get_1d_arrays(t, qt)
    return _jax_sigmoid(t, qt, qfunc_k, qfunc_ylo, qfunc_yhi)


def _jax_quenching_function(t, qt):
    return _jax_sigmoid(t, qt, 10, 1, 0)


def _jax_log_quenching_function(logt, log_qtime, log_qfrac=-2):
    return _jax_sigmoid(logt, log_qtime, 200, 0, log_qfrac)


def _jax_gradual_quenching(logt, logtq, qspeed):
    z = (logt - logtq) * qspeed
    zq = logtq + 2  # Shift to the 95% point
    return -_jax_triweight_sigmoid_kernel(z - zq)


def central_quenching_time(logmp, percentile, **kwargs):
    """Quenching time of central galaxies.

    In this model, the quenching time decreases with increasing mass,
    such that massive BCGs have earlier quenching times relative to
    centrals of Milky Way mass halos.

    Parameters
    ----------
    logmp : float or ndarray of shape (n, )
        Base-10 log of halo mass at z=0

    percentile : float or ndarray of shape (n, )
        percentile = Prob(< y | logmp) for some halo property y.
        For the median quenching time use percentile = 0.5.

    qt_lgmc : float or ndarray, optional
        Value of log10(Mhalo) of the inflection point of qtime

    qt_k : float or ndarray, optional
        Steepness of the qtime sigmoid

    qt_dwarfs : float or ndarray, optional
        Quenching time of dwarf-mass centrals

    qt_clusters : float or ndarray, optional
        Quenching time of cluster-mass centrals

    qt_scatter_lgmc : float or ndarray, optional
        Value of log10(Mhalo) of the inflection point of qtime scatter

    qt_scatter_k : float or ndarray, optional
        Steepness of the qtime scatter sigmoid

    qt_scatter_dwarfs : float or ndarray, optional
        Quenching time scatter in dwarf-mass centrals

    qt_scatter_clusters : float or ndarray, optional
        Quenching time scatter in cluster-mass centrals

    Returns
    -------
    qtime : float or ndarray of shape (n, )
        Age of the universe at the time of quenching in units of Gyr

    """
    logmp, percentile = _get_1d_arrays(logmp, percentile)
    param_dict = _get_default_quenching_time_param_dict(**kwargs)
    params = tuple(param_dict.values())
    qtime = central_quenching_time_jax(logmp, percentile, params)
    return np.asarray(qtime)


def inverse_central_quenching_time(logmp, qthresh, **kwargs):
    """Calculate Prob(qtime < qthresh | logmp) for central galaxies.

    In this model, the quenching time decreases with increasing mass,
    such that massive BCGs have earlier quenching times relative to
    centrals of Milky Way mass halos.

    Parameters
    ----------
    logmp : float or ndarray of shape (n, )
        Base-10 log of halo mass at z=0

    qthresh : float or ndarray of shape (n, )
        Prob(qtime < qthresh | logmp).
        For the median quenching time use percentile = 0.5.

    qt_lgmc : float or ndarray, optional
        Value of log10(Mhalo) of the inflection point of qtime

    qt_k : float or ndarray, optional
        Steepness of the qtime sigmoid

    qt_dwarfs : float or ndarray, optional
        Quenching time of dwarf-mass centrals

    qt_clusters : float or ndarray, optional
        Quenching time of cluster-mass centrals

    qt_scatter_lgmc : float or ndarray, optional
        Value of log10(Mhalo) of the inflection point of qtime scatter

    qt_scatter_k : float or ndarray, optional
        Steepness of the qtime scatter sigmoid

    qt_scatter_dwarfs : float or ndarray, optional
        Quenching time scatter in dwarf-mass centrals

    qt_scatter_clusters : float or ndarray, optional
        Quenching time scatter in cluster-mass centrals

    Returns
    -------
    percentile : float or ndarray of shape (n, )
        percentile = Prob(qtime < qthresh | logmp).
        When qthresh is the median qtime for a halo of mass logmp,
        function returns 0.5

    """
    logmp, qthresh = _get_1d_arrays(logmp, qthresh)
    param_dict = _get_default_quenching_time_param_dict(**kwargs)
    params = tuple(param_dict.values())
    percentile = inverse_central_quenching_time_jax(logmp, qthresh, params)
    return np.asarray(percentile)


def _central_quenching_time_kern(logmp, percentile, params):
    qt_params, qt_scatter_params = params[0:4], params[4:]
    logtq_med = jax_np.log10(_median_quenching_time_kern(logmp, qt_params))
    logtq_scale = _quenching_time_scatter_kern(logmp, qt_scatter_params)
    ylo, yhi = logtq_med - logtq_scale, logtq_med + logtq_scale
    z = _z_score_from_percentile(percentile)
    log_qt = _jax_sigmoid(z, 0, 1, ylo, yhi)
    return jax_np.power(10, log_qt)


def _inverse_central_quenching_time_kern(logmp, qtime, params):
    qt_params, qt_scatter_params = params[0:4], params[4:]
    log_qt = jax_np.log10(qtime)

    log_qt_med = jax_np.log10(_median_quenching_time_kern(logmp, qt_params))
    log_tq_scale = _quenching_time_scatter_kern(logmp, qt_scatter_params)
    ylo, yhi = log_qt_med - log_tq_scale, log_qt_med + log_tq_scale

    qtime_z_score = jax_inverse_sigmoid(log_qt, 0, 1, ylo, yhi)
    qtime_percentile = _percentile_from_z_score(qtime_z_score)
    return qtime_percentile


central_quenching_time_jax = jax_jit(
    jax_vmap(_central_quenching_time_kern, in_axes=(0, 0, None))
)


inverse_central_quenching_time_jax = jax_jit(
    jax_vmap(_inverse_central_quenching_time_kern, in_axes=(0, 0, None))
)


def satellite_quenching_time(logmp, percentile, infall_time, **kwargs):
    """Quenching time of satellite galaxies.

    Minimum of infall time and the corresponding central_quenching_time.

    """
    qtime_cens = central_quenching_time(logmp, percentile, **kwargs)
    return jax_np.minimum(qtime_cens, infall_time)


def _median_quenching_time_kern(logmp, params):
    qt_lgmc, qt_k, qt_dwarfs, qt_clusters = params
    return _jax_sigmoid(logmp, qt_lgmc, qt_k, qt_dwarfs, qt_clusters)


def _quenching_time_scatter_kern(logmp, params):
    qt_scatter_lgmc, qt_scatter_k, qt_scatter_dwarfs, qt_scatter_clusters = params
    return _jax_sigmoid(
        logmp, qt_scatter_lgmc, qt_scatter_k, qt_scatter_dwarfs, qt_scatter_clusters
    )


median_quenching_time_jax = jax_jit(
    jax_vmap(_median_quenching_time_kern, in_axes=(0, None))
)

quenching_time_scatter_jax = jax_jit(
    jax_vmap(_quenching_time_scatter_kern, in_axes=(0, None))
)


@jax_jit
def _z_score_from_percentile(percentile):
    return jax_np.sqrt(2) * jax_erfinv(2 * percentile - 1)


@jax_jit
def _percentile_from_z_score(z_score):
    return 0.5 * (1 + jax_erf(z_score / np.sqrt(2)))


@jax_jit
def _weighted_mixture_of_two_gaussians(g1, g2, r):
    return r * g1 + jax_np.sqrt(1 - r * r) * g2


@jax_jit
def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length."""
    results = [jax_np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [jax_np.zeros(npts).astype(arr.dtype) + arr for arr in results]


def _get_default_quenching_time_param_dict(**kwargs):
    """
    """
    _enforce_no_extraneous_keywords(DEFAULT_PARAMS, **kwargs)

    param_dict = OrderedDict()
    for key, default_value in DEFAULT_PARAMS.items():
        param_dict[key] = kwargs.get(key, default_value)
    return param_dict
