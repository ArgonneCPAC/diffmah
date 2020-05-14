import numpy as np
from jax import numpy as jax_np
from collections import OrderedDict


def get_1d_arrays(*args, jax_arrays=False):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [jax_np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)

    if jax_arrays:
        result = [jax_np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    else:
        result = [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    return result


def jax_sigmoid(x, x0, k, ylo, yhi):
    """Sigmoid function implemented w/ `jax.numpy.exp`.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the function.
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ylo : float or array-like
        The value as x goes to -infty.
    yhi : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input
    """
    return ylo + (yhi - ylo) / (1 + jax_np.exp(-k * (x - x0)))


def jax_inverse_sigmoid(y, x0, k, ylo, yhi):
    """Sigmoid function implemented w/ `jax.numpy.exp`.

    Parameters
    ----------
    y : float or array-like
        Value of the function
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ylo : float or array-like
        The value as x goes to -infty.
    yhi : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input
    """
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jax_np.log(lnarg) / k


def _enforce_no_extraneous_keywords(defaults, **kwargs):
    unrecognized_params = set(kwargs) - set(defaults)

    if len(unrecognized_params) > 0:
        param = list(unrecognized_params)[0]
        msg = (
            "Unrecognized parameter ``{0}``"
            " passed to central_quenching_time function"
        )
        raise KeyError(msg.format(param))


def _get_param_dict(defaults, strict=False, **kwargs):
    """
    """
    param_dict = OrderedDict(
        [(key, kwargs.get(key, val)) for key, val in defaults.items()]
    )
    if strict:
        _enforce_no_extraneous_keywords(defaults, **kwargs)
    return param_dict


def _get_param_array(defaults, strict=False, dtype="f4", jax_arrays=True, **kwargs):
    """
    """
    param_dict = _get_param_dict(defaults, strict=strict, **kwargs)
    if jax_arrays:
        param_array = jax_np.array(list(param_dict.values())).astype(dtype)
    else:
        param_array = np.array(list(param_dict.values())).astype(dtype)
    return param_array
