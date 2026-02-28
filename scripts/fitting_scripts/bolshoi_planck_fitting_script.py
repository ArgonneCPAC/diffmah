""""""

import argparse
import os
import subprocess
from time import time

import h5py
import numpy as np
from mpi4py import MPI

from diffmah.data_loaders import load_bpl_um_data
from diffmah.fitting_helpers import diffmah_fitter_helpers as cfh

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"


DT_FIT_MIN = 0.5  # Minimum halo lifetime in Gyr to bother running fitter


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Root directory storing MAH data")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("-istart", help="First subvolume", type=int, default=0)
    parser.add_argument("-iend", help="Last subvolume", type=int, default=0)

    parser.add_argument(
        "-outbn", help="Basename of the output hdf5 file", default="diffmah_fits.h5"
    )

    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument(
        "-dt_fit_min",
        help="Minimum halo lifetime in Gyr to run fitter",
        default=DT_FIT_MIN,
        type=float,
    )

    args = parser.parse_args()
    indir = args.indir
    outdir = os.path.abspath(args.outdir)
    outbn = args.outbn
    is_test = args.test
    dt_fit_min = args.dt_fit_min
    istart = args.istart
    iend = args.iend

    os.makedirs(outdir, exist_ok=True)

    subvolumes = list(range(istart, iend + 1))
    _mah_data = load_bpl_um_data.load_bpl_mah_data(args.indir, subvolumes)
    halo_ids, mahs, bpl_t_table, log_mah_fit_min, lgt0 = _mah_data
    t0 = 10**lgt0
    # Ensure the target MAHs are cumulative peak masses
    mahs = np.maximum.accumulate(mahs, axis=1)

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 5
    else:
        nhalos_tot = len(halo_ids)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]

    halo_ids_for_rank = halo_ids[indx]
    mahs_for_rank = mahs[indx]
    nhalos_for_rank = len(halo_ids_for_rank)
    nhalos_for_rank = len(halo_ids_for_rank)

    rank_basepat = TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, TMP_OUTPAT).format(rank)

    start = time()

    with open(rank_outname, "w") as fout:
        fout.write(cfh.HEADER)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            mah_i = mahs[i, :]

            fit_results = cfh.diffmah_fitter(
                bpl_t_table,
                mah_i,
                lgm_min=log_mah_fit_min,
                t_fit_min=load_bpl_um_data.BPL_T_FIT_MIN,
                nstep=200,
                n_warmup=1,
            )

            outline = cfh.get_outline(fit_results)
            fout.write(outline)

    comm.Barrier()
    end = time()

    msg = "\n\nWallclock runtime to fit {0} halos with {1} ranks = {2:.1f} seconds\n\n"
    if rank == 0:
        runtime = end - start
        print(msg.format(nhalos_tot, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(args.outdir, rank_basepat)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        collector = []
        for fit_fn in fit_data_fnames:
            assert os.path.isfile(fit_fn)
            fit_data = np.genfromtxt(fit_fn, dtype="str")
            collector.append(fit_data)
        chunk_fit_results = np.concatenate(collector)

        fit_data_bnames = [os.path.basename(fn) for fn in fit_data_fnames]
        outfn = os.path.join(outdir, outbn)

        cfh.write_collated_data(outfn, chunk_fit_results)

        # clean up ASCII data for subvol_i
        bn = fit_data_bnames[0]
        bnpat = "_".join(bn.split("_")[:-1]) + "_*.dat"
        fnpat = os.path.join(outdir, bnpat)
        command = "rm " + fnpat
        subprocess.os.system(command)
