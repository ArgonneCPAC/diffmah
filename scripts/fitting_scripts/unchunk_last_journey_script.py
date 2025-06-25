"""Script to collate the diffmah fits of Last Journey"""

import argparse
import gc
import os
from time import time

import h5py
import numpy as np

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LJ_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_LCRC_DIFFMAH = "/lcrc/project/halotools/LastJourney/diffmah_fits"
DRN_LJ_LCRC_DIFFMAH_OUT = "/lcrc/project/halotools/LastJourney/diffmah_fits_unchunked"

BNPAT_DIFFMAH_FITS = "subvol_{0}_chunk_{1}.hdf5"
BNPAT_DIFFMAH_OUT = "subvol_{0}_diffmah_fits.hdf5"

NCHUNKS = 20
NUM_SUBVOLS_LJ = 192


def _load_flat_hdf5(fn):
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("istart", help="First subvolume", type=int)
    parser.add_argument("iend", help="last subvolume", type=int)

    parser.add_argument(
        "-indir", help="Drn of diffmah fits", default=DRN_LJ_LCRC_DIFFMAH
    )
    parser.add_argument(
        "-outdir",
        help="output drn of collated diffmah fits",
        default=DRN_LJ_LCRC_DIFFMAH_OUT,
    )
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir

    istart = args.istart
    iend = args.iend
    n_subvols = iend - istart

    os.makedirs(outdir, exist_ok=True)

    nchar_chunks = len(str(NCHUNKS))

    start_script = time()
    for isubvol in range(istart, iend):
        start_subvol = time()
        print(f"...collating subvolume {isubvol}")

        subvol_str = f"{isubvol}"

        subvol_collector = []
        for chunknum in range(NCHUNKS):
            chunknum_str = f"{chunknum:0{nchar_chunks}d}"

            bname_in = BNPAT_DIFFMAH_FITS.format(subvol_str, chunknum_str)

            fname_in = os.path.join(indir, bname_in)

            chunk_data = _load_flat_hdf5(fname_in)
            subvol_collector.append(chunk_data)

        data_out = dict()
        colnames = list(subvol_collector[0].keys())
        for key in colnames:
            seq = [x[key] for x in subvol_collector]
            data_out[key] = np.concatenate(seq)
        bname_out = BNPAT_DIFFMAH_OUT.format(subvol_str)
        fname_out = os.path.join(outdir, bname_out)
        with h5py.File(fname_out, "w") as hdfout:
            for key in colnames:
                hdfout[key] = data_out[key]

        del subvol_collector
        del data_out
        gc.collect()
        end_subvol = time()
        runtime_subvol = end_subvol - start_subvol
        msg = f"Runtime for subvol {isubvol} = {runtime_subvol:.2f} seconds"
        print(msg)

    end_script = time()
    script_runtime = end_script - start_script
    msg = f"Total runtime for {n_subvols} = {script_runtime:.2f} seconds"
    print(msg)
