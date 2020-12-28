"""
"""
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from diffmah.individual_halo_history import _calc_halo_history_uparams
from diffmah.galaxy_quenching import _quenching_kern


LGMC_PARAMS = OrderedDict(lgmc_x0=0.735, lgmc_k=5.0, lgmc_ylo=12.3, lgmc_yhi=11.55)
LGNC_PARAMS = OrderedDict(lgnc_x0=1.03, lgnc_k=10, lgnc_ylo=-0.25, lgnc_yhi=-0.8)
LGBD_PARAMS = OrderedDict(lgbd_x0=1.01, lgbd_k=30.0, lgbd_ylo=0.1, lgbd_yhi=-0.035)
LGBC_PARAMS = OrderedDict(lgbc=-0.1)

DEFAULT_PARAMS = deepcopy(LGMC_PARAMS)
DEFAULT_PARAMS.update(LGNC_PARAMS)
DEFAULT_PARAMS.update(LGBD_PARAMS)
DEFAULT_PARAMS.update(LGBC_PARAMS)

LGMC_PARAM_BOUNDS = OrderedDict(
    lgmc_x0=(0.05, 1.25),
    lgmc_k=(1.0, 15.0),
    lgmc_ylo=(10.25, 14.5),
    lgmc_yhi=(10.25, 14.5),
)

LGNC_PARAM_BOUNDS = OrderedDict(
    lgnc_x0=(0.05, 1.25),
    lgnc_k=(1.0, 15.0),
    lgnc_ylo=(-1.0, 0.0),
    lgnc_yhi=(-1.0, 0.0),
)
LGBD_PARAM_BOUNDS = OrderedDict(
    lgbd_x0=(0.05, 1.25),
    lgbd_k=(1.0, 50.0),
    lgbd_ylo=(-0.2, 0.3),
    lgbd_yhi=(-0.2, 0.3),
)

LGBC_PARAM_BOUNDS = OrderedDict(lgbc=(-0.4, 0.0))
DEFAULT_BOUNDS = deepcopy(LGMC_PARAM_BOUNDS)
DEFAULT_BOUNDS.update(LGNC_PARAM_BOUNDS)
DEFAULT_BOUNDS.update(LGBD_PARAM_BOUNDS)
DEFAULT_BOUNDS.update(LGBC_PARAM_BOUNDS)

FB = 0.156


@jjit
def _sm_history(
    lgt, dt, logtmp, logmp, u_x0, u_k, u_lge, u_dy, sfr_ms_params, q_params
):
    _a = lgt, logtmp, logmp, u_x0, u_k, u_lge, u_dy, sfr_ms_params, q_params
    dmhdt, log_mah, main_sequence_sfr, sfr = _sfr_history(*_a)
    mstar = _integrate_sfr(sfr, dt)
    return dmhdt, log_mah, main_sequence_sfr, sfr, mstar


@jjit
def _sfr_history(lgt, logtmp, logmp, u_x0, u_k, u_lge, u_dy, sfr_ms_params, q_params):
    _a = lgt, logtmp, logmp, u_x0, u_k, u_lge, u_dy, sfr_ms_params
    dmhdt, log_mah, main_sequence_sfr = _main_sequence_history(*_a)
    qfrac = _quenching_kern(lgt, *q_params)
    sfr = qfrac * main_sequence_sfr
    return dmhdt, log_mah, main_sequence_sfr, sfr


@jjit
def _main_sequence_history(lgt, logtmp, logmp, u_x0, u_k, u_lge, u_dy, sfr_ms_params):
    dmhdt, log_mah = _calc_halo_history_uparams(
        lgt, logtmp, logmp, u_x0, u_k, u_lge, u_dy
    )
    efficiency = 10 ** _log_ms_eff_u_kern(logmp, lgt, *sfr_ms_params)
    main_sequence_sfr = FB * efficiency * dmhdt
    return dmhdt, log_mah, main_sequence_sfr


@jjit
def _integrate_sfr(sfr, dt):
    return jnp.cumsum(sfr * dt) * 1e9


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0=0, k=1, ymin=-1, ymax=1):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _rolling_plaw_kern(lgx, lgx0, lgy_at_x0, indx_lo, indx_hi):
    slope = _sigmoid(lgx, lgx0, 3, indx_lo, indx_hi)
    return lgy_at_x0 + slope * (lgx - lgx0)


@jjit
def _log_ms_eff_u_kern(
    lgm,
    lgt,
    u_lgmc_x0,
    u_lgmc_k,
    u_lgmc_ylo,
    u_lgmc_yhi,
    u_lgnc_x0,
    u_lgnc_k,
    u_lgnc_ylo,
    u_lgnc_yhi,
    u_lgbd_x0,
    u_lgbd_k,
    u_lgbd_ylo,
    u_lgbd_yhi,
    u_lgbc,
):
    bounded_params = _get_bounded_params(
        u_lgmc_x0,
        u_lgmc_k,
        u_lgmc_ylo,
        u_lgmc_yhi,
        u_lgnc_x0,
        u_lgnc_k,
        u_lgnc_ylo,
        u_lgnc_yhi,
        u_lgbd_x0,
        u_lgbd_k,
        u_lgbd_ylo,
        u_lgbd_yhi,
        u_lgbc,
    )
    return _log_ms_eff_kern(lgm, lgt, bounded_params)


@jjit
def _log_ms_eff_kern(lgm, lgt, bounded_params):
    plaw_params = _get_plaw_params(lgt, bounded_params)
    eff = _rolling_plaw_kern(lgm, *plaw_params)
    return eff


@jjit
def _get_plaw_params(lgt, bounded_params):
    lgmc = _lgmc_vs_lgt(lgt, *bounded_params[0:4])
    lg_eff_at_mc = _lgnc_vs_lgt(lgt, *bounded_params[4:8])
    lg_eff_dwarfs = _lgbd_vs_lgt(lgt, *bounded_params[8:12])
    eff_dwarfs = 10 ** lg_eff_dwarfs
    eff_clusters = -(10 ** bounded_params[12])
    plaw_params = lgmc, lg_eff_at_mc, eff_dwarfs, eff_clusters
    return plaw_params


@jjit
def _get_bounded_params(
    u_lgmc_x0,
    u_lgmc_k,
    u_lgmc_ylo,
    u_lgmc_yhi,
    u_lgnc_x0,
    u_lgnc_k,
    u_lgnc_ylo,
    u_lgnc_yhi,
    u_lgbd_x0,
    u_lgbd_k,
    u_lgbd_ylo,
    u_lgbd_yhi,
    u_lgbc,
):
    lgmc_p = _get_lgmc_bounded_params(u_lgmc_x0, u_lgmc_k, u_lgmc_ylo, u_lgmc_yhi)
    lgnc_p = _get_lgnc_bounded_params(u_lgnc_x0, u_lgnc_k, u_lgnc_ylo, u_lgnc_yhi)
    lgbd_p = _get_lgbd_bounded_params(u_lgbd_x0, u_lgbd_k, u_lgbd_ylo, u_lgbd_yhi)
    lgbc = _get_lgbc_bounded_params(u_lgbc)
    bounded_params = (*lgmc_p, *lgnc_p, *lgbd_p, lgbc)
    return bounded_params


@jjit
def _get_lgmc_bounded_params(u_lgmc_x0, u_lgmc_k, u_lgmc_ylo, u_lgmc_yhi):
    lgmc_x0 = _sigmoid(u_lgmc_x0, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_x0"])
    lgmc_k = _sigmoid(u_lgmc_k, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_k"])
    lgmc_ylo = _sigmoid(u_lgmc_ylo, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_ylo"])
    lgmc_yhi = _sigmoid(u_lgmc_yhi, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_yhi"])
    return lgmc_x0, lgmc_k, lgmc_ylo, lgmc_yhi


@jjit
def _get_lgmc_unbounded_params(lgmc_x0, lgmc_k, lgmc_ylo, lgmc_yhi):
    u_lgmc_x0 = _inverse_sigmoid(lgmc_x0, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_x0"])
    u_lgmc_k = _inverse_sigmoid(lgmc_k, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_k"])
    u_lgmc_ylo = _inverse_sigmoid(lgmc_ylo, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_ylo"])
    u_lgmc_yhi = _inverse_sigmoid(lgmc_yhi, 0, 0.1, *LGMC_PARAM_BOUNDS["lgmc_yhi"])
    return u_lgmc_x0, u_lgmc_k, u_lgmc_ylo, u_lgmc_yhi


@jjit
def _get_lgnc_bounded_params(u_lgnc_x0, u_lgnc_k, u_lgnc_ylo, u_lgnc_yhi):
    lgnc_x0 = _sigmoid(u_lgnc_x0, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_x0"])
    lgnc_k = _sigmoid(u_lgnc_k, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_k"])
    lgnc_ylo = _sigmoid(u_lgnc_ylo, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_ylo"])
    lgnc_yhi = _sigmoid(u_lgnc_yhi, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_yhi"])
    return lgnc_x0, lgnc_k, lgnc_ylo, lgnc_yhi


@jjit
def _get_lgnc_unbounded_params(lgnc_x0, lgnc_k, lgnc_ylo, lgnc_yhi):
    u_lgnc_x0 = _inverse_sigmoid(lgnc_x0, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_x0"])
    u_lgnc_k = _inverse_sigmoid(lgnc_k, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_k"])
    u_lgnc_ylo = _inverse_sigmoid(lgnc_ylo, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_ylo"])
    u_lgnc_yhi = _inverse_sigmoid(lgnc_yhi, 0, 0.1, *LGNC_PARAM_BOUNDS["lgnc_yhi"])
    return u_lgnc_x0, u_lgnc_k, u_lgnc_ylo, u_lgnc_yhi


@jjit
def _get_lgbd_bounded_params(u_lgbd_x0, u_lgbd_k, u_lgbd_ylo, u_lgbd_yhi):
    lgbd_x0 = _sigmoid(u_lgbd_x0, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_x0"])
    lgbd_k = _sigmoid(u_lgbd_k, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_k"])
    lgbd_ylo = _sigmoid(u_lgbd_ylo, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_ylo"])
    lgbd_yhi = _sigmoid(u_lgbd_yhi, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_yhi"])
    return lgbd_x0, lgbd_k, lgbd_ylo, lgbd_yhi


@jjit
def _get_lgbd_unbounded_params(lgbd_x0, lgbd_k, lgbd_ylo, lgbd_yhi):
    u_lgbd_x0 = _inverse_sigmoid(lgbd_x0, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_x0"])
    u_lgbd_k = _inverse_sigmoid(lgbd_k, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_k"])
    u_lgbd_ylo = _inverse_sigmoid(lgbd_ylo, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_ylo"])
    u_lgbd_yhi = _inverse_sigmoid(lgbd_yhi, 0, 0.1, *LGBD_PARAM_BOUNDS["lgbd_yhi"])
    return u_lgbd_x0, u_lgbd_k, u_lgbd_ylo, u_lgbd_yhi


@jjit
def _get_lgbc_bounded_params(u_lgbc):
    lgbc = _sigmoid(u_lgbc, 0, 0.1, *LGBC_PARAM_BOUNDS["lgbc"])
    return lgbc


@jjit
def _get_lgbc_unbounded_params(lgbc):
    u_lgbc = _inverse_sigmoid(lgbc, 0, 0.1, *LGBC_PARAM_BOUNDS["lgbc"])
    return u_lgbc


@jjit
def _lgmc_vs_lgt(lgt, lgmc_x0, lgmc_k, lgmc_ylo, lgmc_yhi):
    lgmc = _sigmoid(lgt, lgmc_x0, lgmc_k, lgmc_ylo, lgmc_yhi)
    return lgmc


@jjit
def _lgnc_vs_lgt(lgt, lgnc_x0, lgnc_k, lgnc_ylo, lgnc_yhi):
    lg_eff_at_mc = _sigmoid(lgt, lgnc_x0, lgnc_k, lgnc_ylo, lgnc_yhi)
    return lg_eff_at_mc


@jjit
def _lgbd_vs_lgt(lgt, lgbd_x0, lgbd_k, lgbd_ylo, lgbd_yhi):
    lg_eff_dwarfs = _sigmoid(lgt, lgbd_x0, lgbd_k, lgbd_ylo, lgbd_yhi)
    return lg_eff_dwarfs


_b = (0, None, 0)
_sfr_eff_halopop = jvmap(_log_ms_eff_kern, in_axes=_b)


def sfr_efficiency_halopop(t_table, log_mah, *u_sfr_params):
    t_table = np.atleast_1d(t_table).astype("f4")
    log_mah = np.atleast_2d(log_mah).astype("f4")
    n_halos, n_t = log_mah.shape
    msg = "z_table shape {0} must agree with mhalo_at_z shape {1}"
    assert n_t == t_table.size, msg.format(t_table.shape, log_mah.shape)
    assert len(u_sfr_params) == len(DEFAULT_PARAMS)
    param_sizes = [p.size for p in u_sfr_params]
    assert np.allclose(param_sizes, n_halos), (param_sizes, n_halos)
    sfr_params = _get_bounded_params(*u_sfr_params)
    return np.array(_sfr_eff_halopop(log_mah, t_table, sfr_params))
