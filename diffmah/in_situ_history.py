"""Module implements in_situ_stellar_mass_at_zobs function to calculate M*(z)."""
import numpy as np
from collections import OrderedDict
from scipy.integrate import trapz, cumtrapz
from astropy.cosmology import Planck15
from .moster17_efficiency import sfr_efficiency_function
from .sigmoid_mah import logmpeak_from_logt, _median_mah_sigmoid_params, LOGT0
from .sigmoid_mah import _logtc_from_mah_percentile, DEFAULT_MAH_PARAMS
from .moster17_efficiency import DEFAULT_PARAMS as DEFAULT_SFR_PARAMS
from .quenching_times import DEFAULT_PARAMS as DEFAULT_QTIME_PARAMS
from .quenching_times import quenching_function, central_quenching_time


Z_INTERP_TABLE = np.linspace(15, -0.25, 5000)
T_INTERP_TABLE = Planck15.age(Z_INTERP_TABLE).value

TARR = np.linspace(0.05, Planck15.age(0).value + 1, 200)
ZARR = np.interp(TARR, T_INTERP_TABLE, Z_INTERP_TABLE)

FB = Planck15.Ob0 / Planck15.Om0


def in_situ_galaxy_halo_history(
    logm0,
    qtime=None,
    t_table=TARR,
    z_table=ZARR,
    fb=FB,
    logt0=LOGT0,
    logtc=None,
    logtk=None,
    dlogm_height=None,
    mah_percentile=None,
    return_mah=False,
    qtime_percentile=0.5,
    **model_params,
):
    """Integrate star formation history to calculate M* at zobs.

    Parameters
    ----------
    logm0 : float
        Base-10 log of halo mass at z=0

    qtime : float, optional
        Quenching time in Gyr. When not set explicitly,
        default qtime is set by the default values in model_params

    t_table : ndarray, optional
        Array of cosmic times used as control points to define the SFH integrand.
        Must be consistent with z_table.

    z_table : ndarray, optional
        Array of redshifts used as control points to define the SFH integrand.
        Must be consistent with t_table.

    fb : float, optional
        Cosmic baryon fraction

    logt0 : float, optional
        Base-10 log of present-day cosmic time.
        Default is 1.14 for the age of a Planck15-Universe at z=0.

    logtc : float, optional
        Base-10 log of the MAH critical time in Gyr.
        Smaller values of logtc produce halos with earlier formation times.
        Default is set by DEFAULT_MAH_PARAMS.
        Since logtc is determined by mah_percentile, and conversely,
        these two parameters may not be specified concurrently.

    logtk : float, optional
        Steepness of transition from fast- to slow-accretion regimes.
        Larger values of k produce quicker-transitioning halos.
        Default is set by DEFAULT_MAH_PARAMS.

    dlogm_height : float
        Total gain in logmpeak until logt0.
        Default is set by DEFAULT_MAH_PARAMS.

    mah_percentile : float, optional
        Value in the interval [0, 1] specifying whether
        the halo is early-forming or late-forming for its mass.
        mah_percentile = 0 <==> early-forming halo
        mah_percentile = 1 <==> late-forming halo
        Default is 0.5 for a halo with a typical assembly history,
        in which case the median logtc value is used.
        Since logtc is determined by mah_percentile, and conversely,
        these two parameters may not be specified concurrently.

    qtime_percentile : float, optional
        Value in the interval [0, 1] specifying whether
        the quenching time is earlier (0) or later (1) relative to
        the quenching time of other quenched galaxies of the same mass.
        Default is 0.5 for median quenching times.

    **model_params : float, optional
        Any parameter regulating main-sequence SFR or quenching times is accepted.

    Returns
    -------
    mah : ndarray, optional
        Array of shape (n, ), where n is t_table.size,
        storing halo masses used when integrating SFR history
        to give mstar at zobs. The value of halo mass at zobs
        is given by mah[-1].

    """
    res = _process_args(
        logm0,
        z_table,
        t_table,
        logtc,
        logtk,
        dlogm_height,
        model_params,
        qtime,
        mah_percentile,
        qtime_percentile,
    )
    zarr, logtarr, logtc, logtk, dlogm_height = res[:5]
    ms_params, qtime, mah_percentile = res[5:]

    tarr, mah, dmdt = _get_mah_history(
        zarr, logtarr, logtc, logtk, dlogm_height, logm0, logt0
    )

    epsilon = sfr_efficiency_function(mah, zarr, **ms_params)

    sfr_ms_history = fb * dmdt * epsilon
    sfr_q_history = sfr_ms_history * quenching_function(tarr, qtime)

    _mstar_ms_history = cumtrapz(sfr_ms_history, x=tarr)
    _mstar_q_history = cumtrapz(sfr_q_history, x=tarr)
    mstar_ms_history = np.insert(_mstar_ms_history, 0, _mstar_ms_history[0])
    mstar_q_history = np.insert(_mstar_q_history, 0, _mstar_q_history[0])

    return (
        zarr,
        tarr,
        mah,
        dmdt,
        sfr_ms_history,
        sfr_q_history,
        mstar_ms_history,
        mstar_q_history,
    )


def _get_mah_history(zarr, logtarr, logtc, logtk, dlogm_height, logm0, logt0):
    mah = 10 ** logmpeak_from_logt(logtarr, logtc, logtk, dlogm_height, logm0, logt0)

    tarr = 10 ** logtarr
    _dmdt = np.diff(mah) / np.diff(tarr)
    dmdt = np.insert(_dmdt, 0, _dmdt[0])

    return tarr, mah, dmdt


def _process_args(
    logm0,
    z_table,
    t_table,
    logtc,
    logtk,
    dlogm_height,
    model_params,
    qtime,
    mah_percentile,
    qtime_percentile,
):
    zobs = 0
    assert np.shape(logm0) == (), "logm0 should be a float"
    msg = "Must have {0} <= zobs <= {1}".format(z_table.min(), z_table.max())
    assert z_table.min() <= zobs <= z_table.max(), msg
    zarr = np.linspace(z_table.max(), zobs, z_table.size)
    tarr = np.interp(zarr, z_table[::-1], t_table[::-1])
    logtarr = np.log10(tarr)

    defaults = list((DEFAULT_MAH_PARAMS, DEFAULT_SFR_PARAMS, DEFAULT_QTIME_PARAMS,))
    param_list = _get_model_param_dictionaries(*defaults, **model_params)
    mah_params, ms_params, qtime_params = param_list

    inconsistent = (mah_percentile is not None) & (logtc is not None)
    try:
        assert not inconsistent
    except AssertionError:
        msg = "Do not pass both mah_percentile and logtc"
        raise ValueError(msg)

    logtc_med, logtk_med, dlogm_height_med = _median_mah_sigmoid_params(
        logm0, **mah_params
    )
    if logtc is None:
        if mah_percentile is None:
            logtc = logtc_med
            mah_percentile = 0.5
        else:
            logtc = _logtc_from_mah_percentile(logm0, mah_percentile, **mah_params)
    else:
        msg = "logtc should be a float, instead got {}"
        assert np.shape(logtc) == (), msg.format(logtc)
        logtc_lo = _logtc_from_mah_percentile(logm0, 0, **mah_params)
        logtc_med = _logtc_from_mah_percentile(logm0, 0.5, **mah_params)
        logtc_hi = _logtc_from_mah_percentile(logm0, 1, **mah_params)
        if logtc < logtc_lo:
            mah_percentile = 0
        elif logtc > logtc_hi:
            mah_percentile = 1
        else:
            mah_percentile = (logtc - logtc_lo) / (logtc_hi - logtc_lo)

    if logtk is None:
        logtk = logtk_med
    if dlogm_height is None:
        dlogm_height = dlogm_height_med

    if qtime is None:
        qtime = central_quenching_time(logm0, qtime_percentile, **qtime_params)[0]

    return (
        zarr,
        logtarr,
        logtc,
        logtk,
        dlogm_height,
        ms_params,
        qtime,
        mah_percentile,
    )


def _get_model_param_dictionaries(*default_param_dicts, **input_model_params):

    _s = [list(d.keys()) for d in default_param_dicts]
    all_input_param_names = [key for keylist in _s for key in keylist]

    try:
        assert len(all_input_param_names) == len(set(all_input_param_names))
    except AssertionError:
        raise KeyError("Model parameter names must be unique")

    pat = "Parameter name `{0}` not recognized as a model parameter"
    input_keys = list(input_model_params.keys())
    recognized_keys = all_input_param_names
    unrecognized_inputs = list(set(input_keys) - set(recognized_keys))
    try:
        assert len(unrecognized_inputs) == 0
    except AssertionError:
        raise KeyError(pat.format(unrecognized_inputs[0]))

    model_param_list = []
    for default_param_dict in default_param_dicts:
        model_dict = OrderedDict()
        for param, default_value in default_param_dict.items():
            value = input_model_params.get(param, default_value)
            model_dict[param] = value
        model_param_list.append(model_dict)

    return tuple(model_param_list)
