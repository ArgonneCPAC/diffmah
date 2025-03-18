"""Script to fit Bolshoi, MDPL2, or TNG MAHs with the diffmah model."""

import argparse
import os
import subprocess
from glob import glob
from time import time

import numpy as np
from mpi4py import MPI

from diffmah import fitting_helpers as cfh
from diffmah.data_loaders import load_SMDPL_mahs

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"
BNPAT = "diffmah_fits_subvol_{0}"

NCHUNKS = 20
N_SUBVOL_SMDPL = 576


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Root directory storing SMDPL")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument(
        "-outbase", help="Basename of the output hdf5 file", default="diffmah_fits.h5"
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument(
        "-iend", help="Last subvolume in loop", type=int, default=N_SUBVOL_SMDPL
    )
    parser.add_argument("-nstep", help="Number of gd steps", type=int, default=200)
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)
    parser.add_argument(
        "-num_subvols_tot",
        help="Total # subvols",
        type=int,
        default=N_SUBVOL_SMDPL,
    )

    args = parser.parse_args()
    sim_name = args.sim_name
    nstep = args.nstep
    istart, iend = args.istart, args.iend

    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    nchar_chunks = len(str(nchunks))

    if args.indir == "DR1":
        indir = load_SMDPL_mahs.BEBOP_SMDPL_DR1
    elif args.indir == "DR1_nomerging":
        indir = load_SMDPL_mahs.BEBOP_SMDPL_nomerging
    else:
        indir = args.indir

    os.makedirs(outdir, exist_ok=True)

    start = time()

    if args.test:
        subvolumes = [np.arange(istart, iend, 1).astype(int)[0]]
        chunks = [0, 1, 2]
    else:
        subvolumes = np.arange(istart, iend, 1).astype(int)
        chunks = np.arange(nchunks).astype(int)

    for isubvol in subvolumes:
        isubvol_start = time()

        nchar_subvol = len(str(num_subvols_tot))

        subvol_str = f"{isubvol}"
        bname = BNPAT.format(subvol_str)
        fn_data = os.path.join(indir, bname)

        tarr, mahs = load_SMDPL_mahs.load_SMDPL_data(isubvol, indir)

        nhalos_tot = len(mahs)
        indx_all = np.arange(0, nhalos_tot).astype("i8")
        indx = np.array_split(indx_all, nranks)[rank]

        mahs_for_rank = mahs[indx]

        chunknum = rank

        comm.Barrier()
        ichunk_start = time()

        nhalos_for_rank = mahs_for_rank.shape[0]
        nhalos_tot = comm.reduce(nhalos_for_rank, op=MPI.SUM)

        chunknum_str = f"{chunknum:0{nchar_chunks}d}"
        outbase_chunk = f"subvol_{subvol_str}_chunk_{chunknum_str}"
        rank_basepat = "_".join((outbase_chunk, TMP_OUTPAT))
        rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

        comm.Barrier()
        with open(rank_outname, "w") as fout:
            fout.write(cfh.HEADER)

            for i in range(nhalos_for_rank):
                mah = mahs_for_rank[i, :]
                fit_results = cfh.diffmah_fitter(tarr, mah)
                outline = cfh.get_outline(fit_results)
                fout.write(outline)

        comm.Barrier()
        ichunk_end = time()

        msg = "\n\nWallclock runtime to fit {0} halos with {1} ranks = {2:.1f} seconds\n\n"
        if rank == 0:
            print("\nFinished with subvolume {} chunk {}".format(isubvol, chunknum))
            runtime = ichunk_end - ichunk_start
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

            outbn = BNPAT.format(isubvol) + ".hdf5"
            outfn = os.path.join(outdir, outbn)

            cfh.write_collated_data(outfn, chunk_fit_results)

            # clean up ASCII data for subvol_i
            fit_data_bnames = [os.path.basename(fn) for fn in fit_data_fnames]
            bn = fit_data_bnames[0]
            bnpat = "_".join(bn.split("_")[:-1]) + "_*.dat"
            fnpat = os.path.join(outdir, bnpat)
            command = "rm " + fnpat
            subprocess.os.system(command)
