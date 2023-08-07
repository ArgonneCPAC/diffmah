"""Script to fit Bolshoi, MDPL2, or TNG MAHs with the diffmah model."""
import argparse
import os
import subprocess
from glob import glob
from time import time

import h5py
import numpy as np
from diffmah.fit_mah_helpers import (
    get_header,
    get_loss_data,
    get_outline,
    get_outline_bad_fit,
    log_mah_mse_loss_and_grads,
)
from diffmah.utils import jax_adam_wrapper
from mpi4py import MPI

from load_smdpl import load_smdpl_histories

TMP_OUTPAT = "_tmp_mah_fits_rank_{0}.dat"
TODAY = 13.8

TASSO = os.path.join(
    "/Users/aphearin/work/DATA/SMDPL",
    "dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000",
)
BEBOP = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000"
)


def _write_collated_data(outname, data):
    nrows, ncols = np.shape(data)
    colnames = get_header()[1:].strip().split()
    assert len(colnames) == ncols, "data mismatched with header"
    with h5py.File(outname, "w") as hdf:
        for i, name in enumerate(colnames):
            if name == "halo_id":
                hdf[name] = data[:, i].astype("i8")
            else:
                hdf[name] = data[:, i]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Input directory")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output hdf5 file")
    parser.add_argument("-nstep", help="Num opt steps per halo", type=int, default=200)
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)

    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    rank_basepat = args.outbase + TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)
    nstep = args.nstep

    if args.indir == "TASSO":
        indir = TASSO
    elif args.indir == "BEBOP":
        indir = BEBOP
    else:
        indir = args.indir

    all_subvol_names = [
        os.path.basename(drn) for drn in glob(os.path.join(indir, "subvol_*"))
    ]
    all_subvolumes = [int(s.split("_")[1]) for s in all_subvol_names]
    if args.test:
        subvolumes = [
            all_subvolumes[0],
        ]
    else:
        subvolumes = all_subvolumes

    mock, tarr, lgmh_min = load_smdpl_histories(TASSO, subvolumes)

    # Ensure the target MAHs are cumulative peak masses
    log_mahs = np.maximum.accumulate(mock["log_mah"], axis=1)

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 5
    else:
        nhalos_tot = len(mock["halo_id"])
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]

    halo_ids_for_rank = mock["halo_id"][indx]
    log_mahs_for_rank = log_mahs[indx]
    nhalos_for_rank = len(halo_ids_for_rank)

    header = get_header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            lgmah = log_mahs_for_rank[i, :]

            p_init, loss_data = get_loss_data(
                tarr,
                lgmah,
                lgmh_min,
            )
            _res = jax_adam_wrapper(
                log_mah_mse_loss_and_grads, p_init, loss_data, nstep, n_warmup=1
            )
            p_best, loss_best, loss_arr, params_arr, fit_terminates = _res

            if fit_terminates == 1:
                outline = get_outline(halo_id, loss_data, p_best, loss_best)
            else:
                outline = get_outline_bad_fit(halo_id, lgmah[-1], TODAY)

            fout.write(outline)

    comm.Barrier()
    end = time()

    msg = (
        "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
    )
    if rank == 0:
        runtime = end - start
        print(msg.format(nhalos_tot, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(args.outdir, rank_basepat)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        data_collection = [np.loadtxt(fn) for fn in fit_data_fnames]
        all_fit_data = np.concatenate(data_collection)
        outname = os.path.join(args.outdir, args.outbase)
        _write_collated_data(outname, all_fit_data)

        #  clean up temporary files
        _remove_basename = pat.replace("{0}", "*")
        command = "rm -rf " + _remove_basename
        raw_result = subprocess.check_output(command, shell=True)
