"""
"""
from astropy.table import Table
import h5py
import os
from diffmah.individual_halo_assembly import _get_u_params_from_params_vmap
from diffmah.individual_halo_assembly import _calc_halo_history_vmap
from halotools.sim_manager import CachedHaloCatalog
from halotools.utils import crossmatch
import numpy as np
import warnings
from numba import jit as numba_jit


DRN = "/Users/aphearin/work/DATA/diffmah_data/PENULTIMATE_FITS"


def _load_fits(bn, drn=DRN):
    fn = os.path.join(drn, bn)
    t = Table()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            t[key] = hdf[key][...]
    _keys = ("mah_x0", "mah_k", "early_index", "late_index")

    _up = _get_u_params_from_params_vmap(*(t[key] for key in _keys))
    for key, up in zip(_keys, _up):
        t["u_" + key] = up
    return t


def load_nbody(bolshoi_bn, bolshoi_bn2, mdpl_bn, mdpl2_bn2, drn=DRN):
    bpl, t_bpl = load_bolshoi(bolshoi_bn, bolshoi_bn2, drn=drn)
    mdpl, t_mdpl = load_mdpl2(mdpl_bn, mdpl2_bn2, drn=drn)
    return bpl, mdpl, t_bpl, t_mdpl


def load_bolshoi(bn, bn2, drn=DRN):
    halos = _load_fits(bn, drn=drn)
    halocat = CachedHaloCatalog(simname="bolplanck", redshift=0)
    idxA, idxB = crossmatch(halos["halo_id"], halocat.halo_table["halo_id"])

    keys_to_inherit = (
        "x",
        "y",
        "z",
        "vmax",
        "conc",
        "upid",
        "logm_sim",
        "a_firstacc",
        "tidal_force_tdyn",
    )
    for key in keys_to_inherit:
        halos[key] = 0.0
    halos["x"][idxA] = halocat.halo_table["halo_x"][idxB]
    halos["y"][idxA] = halocat.halo_table["halo_y"][idxB]
    halos["z"][idxA] = halocat.halo_table["halo_z"][idxB]
    halos["vmax"][idxA] = halocat.halo_table["halo_vmax_mpeak"][idxB]
    halos["logm_sim"][idxA] = np.log10(halocat.halo_table["halo_mvir"][idxB])
    halos["conc"][idxA] = halocat.halo_table["halo_nfw_conc"][idxB]
    halos["upid"][idxA] = halocat.halo_table["halo_upid"][idxB]
    halos["a_firstacc"][idxA] = halocat.halo_table["halo_scale_factor_firstacc"][idxB]
    halos["tidal_force_tdyn"][idxA] = halocat.halo_table["halo_tidal_force_tdyn"][idxB]

    orig_fn = os.path.join(os.path.dirname(DRN), bn2)
    orig = Table(np.load(orig_fn))
    t = np.loadtxt(os.path.join(os.path.dirname(DRN), "cosmic_times_bpl.dat"))

    assert np.allclose(halos["halo_id"], orig["halo_id"]), "unexpected mismatch"
    halos["log_mah_sim"] = orig["mpeak_history_main_prog"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zmsk = halos["log_mah_sim"] == 0
        halos["log_mah_sim"] = np.where(zmsk, 0, np.log10(halos["log_mah_sim"]))
    halos["logmp_sim"] = halos["log_mah_sim"][:, -1]

    dmhdt, log_mah = _calc_halo_history_vmap(
        np.log10(t),
        np.log10(halos["tmpeak"]),
        halos["logmp_fit"],
        halos["mah_x0"],
        halos["mah_k"],
        halos["early_index"],
        halos["late_index"],
    )
    halos["dmhdt_fit"] = dmhdt
    halos["log_mah_fit"] = log_mah

    halos["t04_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.04
    )
    halos["t25_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.25
    )
    halos["t50_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.5
    )
    halos["t80_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.8
    )

    halos["t04_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.04
    )
    halos["t25_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.25
    )
    halos["t50_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.5
    )
    halos["t80_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.8
    )

    halos["t04_fit2"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_sim"], t, 0.04
    )
    halos["t25_fit2"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_sim"], t, 0.25
    )
    halos["t50_fit2"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_sim"], t, 0.5
    )
    halos["t80_fit2"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_sim"], t, 0.8
    )

    msk = halos["late_index"] > _sigmoid(halos["early_index"], 3, 1, -0.1, 3.25)
    halos["pop1"] = msk

    return halos, t


def load_mdpl2(bn, bn2, drn=DRN):
    halos = _load_fits(bn, drn=drn)
    halocat = np.load(
        os.path.join(os.path.dirname(DRN), "hlist_1.00000.list.reduced2.npy")
    )
    idxA, idxB = crossmatch(halos["halo_id"], halocat["id"])

    keys_to_inherit = ("x", "y", "z", "vmax", "conc", "upid", "logm_sim")
    for key in keys_to_inherit:
        halos[key] = 0.0
    halos["x"][idxA] = halocat["x"][idxB]
    halos["y"][idxA] = halocat["y"][idxB]
    halos["z"][idxA] = halocat["z"][idxB]
    halos["vmax"][idxA] = halocat["vmax"][idxB]
    halos["logm_sim"][idxA] = np.log10(halocat["mvir"][idxB])
    halos["conc"][idxA] = halocat["rvir"][idxB] / halocat["rs"][idxB]
    halos["upid"][idxA] = halocat["upid"][idxB]

    orig_fn = os.path.join(os.path.dirname(DRN), bn2)
    orig = Table(np.load(orig_fn))
    t = np.loadtxt(os.path.join(os.path.dirname(DRN), "mdpl2_cosmic_time.txt"))

    assert np.allclose(halos["halo_id"], orig["halo_id"]), "unexpected mismatch"
    halos["log_mah_sim"] = orig["mpeak_history_main_prog"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zmsk = halos["log_mah_sim"] == 0
        halos["log_mah_sim"] = np.where(zmsk, 0, np.log10(halos["log_mah_sim"]))
    halos["logmp_sim"] = halos["log_mah_sim"][:, -1]

    dmhdt, log_mah = _calc_halo_history_vmap(
        np.log10(t),
        np.log10(halos["tmpeak"]),
        halos["logmp_fit"],
        halos["mah_x0"],
        halos["mah_k"],
        halos["early_index"],
        halos["late_index"],
    )
    halos["dmhdt_fit"] = dmhdt
    halos["log_mah_fit"] = log_mah

    halos["t04_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.04
    )
    halos["t25_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.25
    )
    halos["t50_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.5
    )
    halos["t80_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.8
    )

    halos["t04_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.04
    )
    halos["t25_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.25
    )
    halos["t50_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.5
    )
    halos["t80_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.8
    )

    msk = halos["late_index"] > _sigmoid(halos["early_index"], 3, 1, -0.1, 3.25)
    halos["pop1"] = msk

    return halos, t


def load_tng(bn, bn2, drn=DRN):
    halos = _load_fits(bn, drn=drn)

    orig_fn = os.path.join(os.path.dirname(DRN), bn2)
    orig = Table(np.load(orig_fn))
    t = np.load(os.path.join(os.path.dirname(DRN), "tn_cosmic_time.npy"))
    halos["xh"] = orig["pos"][:, :, 0]
    halos["yh"] = orig["pos"][:, :, 1]
    halos["zh"] = orig["pos"][:, :, 2]
    msk = orig["cen1_sat0"][:, -1] == 1
    halos["upid"] = np.where(msk, -1, 0)

    halos["log_mah_sim"] = orig["mpeakh"]
    halos["logmp_sim"] = halos["log_mah_sim"][:, -1]

    dmhdt, log_mah = _calc_halo_history_vmap(
        np.log10(t),
        np.log10(halos["tmpeak"]),
        halos["logmp_fit"],
        halos["mah_x0"],
        halos["mah_k"],
        halos["early_index"],
        halos["late_index"],
    )
    halos["dmhdt_fit"] = dmhdt
    halos["log_mah_fit"] = log_mah

    halos["t04_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.04
    )
    halos["t25_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.25
    )
    halos["t50_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.5
    )
    halos["t80_sim"] = _compute_formation_time(
        halos["log_mah_sim"], halos["logmp_sim"], t, 0.8
    )

    halos["t04_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.04
    )
    halos["t25_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.25
    )
    halos["t50_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.5
    )
    halos["t80_fit"] = _compute_formation_time(
        halos["log_mah_fit"], halos["logmp_fit"], t, 0.8
    )

    msk = halos["late_index"] > _sigmoid(halos["early_index"], 3, 1, -0.1, 3.25)
    halos["pop1"] = msk

    return halos, t


def _compute_formation_time(log_mahs, logmps, tarr, f, stnoise=0.05):
    nh, nt = log_mahs.shape
    tmparr = np.zeros(nh)
    for i, log_mah in enumerate(log_mahs):
        y = logmps[i] + np.log10(f)
        tmparr[i] = tarr[find_indx_frac_mass(log_mah, y, nt)]
    return np.random.normal(loc=tmparr, scale=stnoise)


@numba_jit
def find_indx_frac_mass(x, y, n):
    """Find the index where x first attains value y."""
    indx_xpeak = -1
    for i in range(n - 1, -1, -1):
        x_i = x[i]
        if x_i < y:
            indx_xpeak = i
            break
    return indx_xpeak


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))
