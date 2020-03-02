"""Model for the quenching time of central galaxies."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
from jax.scipy.special import erfinv as jax_erfinv
from jax import jit as jax_jit
from jax import vmap as jax_vmap


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


def central_quenching_time(logm0, percentile, **kwargs):
    """Quenching time of central galaxies.

    In this model, the quenching time decreases with increasing mass,
    such that massive BCGs have earlier quenching times relative to
    centrals of Milky Way mass halos.

    Parameters
    ----------
    logm0 : float or ndarray of shape (n, )
        Base-10 log of halo mass at z=0

    percentile : float or ndarray of shape (n, )
        percentile = Prob(< y | logm0) for some halo property y.
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
    logm0, percentile = _get_1d_arrays(logm0, percentile)
    param_dict = _get_default_quenching_time_param_dict(**kwargs)
    params = tuple(param_dict.values())
    qtime = central_quenching_time_jax(logm0, percentile, params)
    return np.asarray(qtime)


def _central_quenching_time_kern(logm0, percentile, params):
    qt_params, qt_scatter_params = params[0:4], params[4:]
    logtq_med = jax_np.log10(_median_quenching_time_kern(logm0, qt_params))
    logtq_scale = _quenching_time_scatter_kern(logm0, qt_scatter_params)
    ylo, yhi = logtq_med - logtq_scale, logtq_med + logtq_scale
    z = _z_score_from_percentile(percentile)
    log_qt = _jax_sigmoid(z, 0, 1, ylo, yhi)
    return jax_np.power(10, log_qt)


central_quenching_time_jax = jax_jit(
    jax_vmap(_central_quenching_time_kern, in_axes=(0, 0, None))
)


def satellite_quenching_time(logm0, percentile, infall_time, **kwargs):
    """Quenching time of satellite galaxies.

    Minimum of infall time and the corresponding central_quenching_time.

    """
    qtime_cens = central_quenching_time(logm0, percentile, **kwargs)
    return jax_np.minimum(qtime_cens, infall_time)


def _median_quenching_time_kern(logm0, params):
    qt_lgmc, qt_k, qt_dwarfs, qt_clusters = params
    return _jax_sigmoid(logm0, qt_lgmc, qt_k, qt_dwarfs, qt_clusters)


def _quenching_time_scatter_kern(logm0, params):
    qt_scatter_lgmc, qt_scatter_k, qt_scatter_dwarfs, qt_scatter_clusters = params
    return _jax_sigmoid(
        logm0, qt_scatter_lgmc, qt_scatter_k, qt_scatter_dwarfs, qt_scatter_clusters
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
def _weighted_mixture_of_two_gaussians(g1, g2, r):
    return r * g1 + jax_np.sqrt(1 - r * r) * g2


@jax_jit
def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length."""
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]


def _enforce_no_extraneous_keywords(**kwargs):
    unrecognized_params = set(kwargs) - set(DEFAULT_PARAMS)

    if len(unrecognized_params) > 0:
        param = list(unrecognized_params)[0]
        msg = (
            "Unrecognized parameter ``{0}``"
            " passed to central_quenching_time function"
        )
        raise KeyError(msg.format(param))


def _get_default_quenching_time_param_dict(**kwargs):
    """
    """
    _enforce_no_extraneous_keywords(**kwargs)

    param_dict = OrderedDict()
    for key, default_value in DEFAULT_PARAMS.items():
        param_dict[key] = kwargs.get(key, default_value)
    return param_dict
