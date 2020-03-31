"""Module implements in_situ_stellar_mass_at_zobs function to calculate M*(z)."""
import numpy as np
from collections import OrderedDict
from scipy.integrate import trapz
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


def in_situ_mstar_at_zobs(
    zobs,
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
    **model_params,
):
    """Integrate star formation history to calculate M* at zobs.

    Parameters
    ----------
    zobs : float

    logm0 : float
        Base-10 log of halo mass at logt0

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
        Base-10 log of cosmic time when the halo attains mass logm0.
        Default is 1.14 for the age of a Planck15-Universe at z=0.

    logtc : float, optional
        Base-10 log of the MAH critical time in Gyr.
        Smaller values of logtc produce halos with earlier formation times.
        Default is set by DEFAULT_MAH_PARAMS.

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
        Default is 0.5 for a halo with a typical assembly history.

    **model_params : float, optional
        Any parameter regulating main-sequence SFR or quenching is accepted.

    Returns
    -------
    mstar_ms : float
        Stellar mass formed in-situ at zobs,
        if the galaxy remained on the main sequence.

    mstar_q : float
        Stellar mass formed in-situ at zobs,
        if the galaxy were quenched at qtime.

    """

    res = _process_args(
        zobs,
        logm0,
        z_table,
        t_table,
        logtc,
        logtk,
        dlogm_height,
        model_params,
        qtime,
        mah_percentile,
    )
    zarr, logtarr, logtc, logtk, dlogm_height = res[:5]
    ms_params, qtime, mah_percentile = res[5:]

    mah = 10 ** logmpeak_from_logt(logtarr, logtc, logtk, dlogm_height, logm0, logt0)

    tarr = 10 ** logtarr
    _dmdt = np.diff(mah) / np.diff(tarr)
    dmdt = np.insert(_dmdt, 0, _dmdt[0])

    epsilon = sfr_efficiency_function(mah, zarr, **ms_params)

    _ms_sfr_integrand = fb * dmdt * epsilon
    _q_sfr_integrand = _ms_sfr_integrand * quenching_function(t_table, qtime)

    mstar_ms = trapz(_ms_sfr_integrand, x=tarr)
    mstar_q = trapz(_q_sfr_integrand, x=tarr)

    return mstar_ms, mstar_q


def _process_args(
    zobs,
    logm0,
    z_table,
    t_table,
    logtc,
    logtk,
    dlogm_height,
    model_params,
    qtime,
    mah_percentile,
):
    assert np.shape(zobs) == (), "zobs should be a float"
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

    logtc_med, logtk_med, dlogm_height_med = _median_mah_sigmoid_params(logm0)
    if logtc is None:
        if mah_percentile is None:
            logtc = logtc_med
            mah_percentile = 0.5
        else:
            logtc = _logtc_from_mah_percentile(logm0, mah_percentile, **mah_params)
    else:
        logtc_lo = _logtc_from_mah_percentile(logm0, 0.5, **mah_params)
        logtc_med = _logtc_from_mah_percentile(logm0, 0.5, **mah_params)
        logtc_hi = _logtc_from_mah_percentile(logm0, 1, **mah_params)
        mah_percentile = (logtc - logtc_lo) / (logtc_hi - logtc_lo)

    if logtk is None:
        logtk = logtk_med
    if dlogm_height is None:
        dlogm_height = dlogm_height_med

    if qtime is None:
        qtime = central_quenching_time(logm0, mah_percentile, **qtime_params)

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
