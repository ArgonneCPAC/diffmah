"""Load the diffmah data into memory."""
import numpy as np
import os
import warnings

TASSO = "/Users/aphearin/work/DATA/diffmah_data/PUBLISHED_DATA"
BEBOP = "/lcrc/project/halotools/diffmah_data/PUBLISHED_DATA"


def load_tng_data(data_drn=BEBOP):
    tng_t = np.load(os.path.join(data_drn, "tng_cosmic_time.npy"))
    fn = os.path.join(data_drn, "tng_diffmah.npy")
    _halos = np.load(fn)
    halo_ids = np.arange(len(_halos["mpeak"])).astype("i8")
    log_mahs = np.maximum.accumulate(_halos["mpeakh"], axis=1)
    log_mah_fit_min = 10.0
    return halo_ids, log_mahs, tng_t, log_mah_fit_min


def load_bolshoi_data(gal_type, data_drn=BEBOP):
    basename = "bpl_diffmah_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    _halos = np.load(fn)
    halo_ids = _halos["halo_id"]

    _mah = np.maximum.accumulate(_halos["mpeak_history_main_prog"], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mahs = np.where(_mah == 0, 0, np.log10(_mah))

    bpl_t = np.load(os.path.join(data_drn, "bpl_cosmic_time.npy"))
    log_mah_fit_min = 10.0
    return halo_ids, log_mahs, bpl_t, log_mah_fit_min


def load_mdpl2_data(gal_type, data_drn=BEBOP):
    basename = "mdpl2_diffmah_{}.npy".format(gal_type)
    fn = os.path.join(data_drn, basename)
    _halos = np.load(fn)
    halo_ids = _halos["halo_id"]

    _mah = np.maximum.accumulate(_halos["mpeak_history_main_prog"], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mahs = np.where(_mah == 0, 0, np.log10(_mah))

    mdpl2_t = np.loadtxt(os.path.join(data_drn, "mdpl2_cosmic_time.txt"))
    log_mah_fit_min = 11.25
    return halo_ids, log_mahs, mdpl2_t, log_mah_fit_min
