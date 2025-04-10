""" """

import h5py

DRN_LCRC_APH2 = "/lcrc/project/halotools/Galacticus/diffstarpop_data/nstep_390"
BNAME_APH2 = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin.hdf5"


def get_nhalos_tot(fn):
    with h5py.File(fn, "r") as hdf:
        nodeData = hdf["Outputs"]["Output1"]["nodeData"]
        nhalos_tot = nodeData["haloAccretionHistoryMass"].size
    return nhalos_tot


def extract_galacticus_mah_tables(fn, istart=0, iend=None):

    with h5py.File(fn, "r") as hdf:
        nodeData = hdf["Outputs"]["Output1"]["nodeData"]

        if iend is None:
            iend = nodeData["haloAccretionHistoryMass"].size

        MAHdata = nodeData["haloAccretionHistoryMass"][istart:iend]
        MAHtimes = nodeData["haloAccretionHistoryTime"][istart:iend]
        nodeIsIsolated = nodeData["nodeIsIsolated"][istart:iend]
        basicTimeLastIsolated = nodeData["basicTimeLastIsolated"][istart:iend]

    return MAHdata, MAHtimes, nodeIsIsolated, basicTimeLastIsolated
