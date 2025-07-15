"""Script to fit Last Journey MAHs with diffmah"""

import argparse
import os
import subprocess
from time import time

import numpy as np
from mpi4py import MPI

from diffmah.data_loaders.load_hacc_mahs import load_mahs_per_rank
from diffmah.fitting_helpers import diffmah_fitter_helpers as cfh

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"

LCRC = None
POBOY_ROOT_DRN = "/Users/aphearin/work/DATA/LastJourney"
LCRC_ROOT_DRN = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"

BNPAT = "m000p.coreforest.{}.hdf5"

NCHUNKS = 20
NUM_SUBVOLS_LASTJOURNEY = 256
SIM_NAME = "LastJourney"


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Input directory")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output hdf5 file")
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument("-iend", help="Last subvolume in loop", type=int, default=-1)
    parser.add_argument("-nstep", help="Number of gd steps", type=int, default=200)
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)
    parser.add_argument(
        "-num_subvols_tot",
        help="Total # subvols",
        type=int,
        default=NUM_SUBVOLS_LASTJOURNEY,
    )

    args = parser.parse_args()
    nstep = args.nstep
    istart, iend = args.istart, args.iend

    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    nchar_chunks = len(str(nchunks))

    os.makedirs(outdir, exist_ok=True)

    if args.indir == "POBOY":
        indir = POBOY_ROOT_DRN
    else:
        indir = args.indir

    start = time()

    all_avail_bnames = [BNPAT.format(i) for i in range(0, num_subvols_tot)]
    all_avail_fnames = [os.path.join(indir, bn) for bn in all_avail_bnames]
    all_avail_subvolumes = [int(s.split(".")[2]) for s in all_avail_bnames]
    all_avail_subvolumes = np.array(sorted(all_avail_subvolumes))

    if args.test:
        subvolumes = [all_avail_subvolumes[0]]
        chunks = [0, 1, 2]
    else:
        subvolumes = all_avail_subvolumes[istart:iend]
        chunks = np.arange(nchunks).astype(int)

    n_subvols = len(subvolumes)
    if rank == 0:
        print(
            f"...Beginning loop over {n_subvols} subvolumes each with {nchunks} chunks"
        )

    for isubvol in subvolumes:
        isubvol_start = time()

        subvol_str = f"{isubvol}"
        bname = BNPAT.format(isubvol)
        fn_data = os.path.join(indir, bname)

        for chunknum in chunks:
            comm.Barrier()
            ichunk_start = time()

            tarr, mahs_for_rank = load_mahs_per_rank(
                fn_data, SIM_NAME, chunknum, nchunks, comm=MPI.COMM_WORLD
            )
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

                fit_data_bnames = [os.path.basename(fn) for fn in fit_data_fnames]
                outbn = "_".join(fit_data_bnames[0].split("_")[:4]) + ".hdf5"
                outfn = os.path.join(outdir, outbn)

                cfh.write_collated_data(outfn, chunk_fit_results)

                # clean up ASCII data for subvol_i
                bn = fit_data_bnames[0]
                bnpat = "_".join(bn.split("_")[:-1]) + "_*.dat"
                fnpat = os.path.join(outdir, bnpat)
                command = "rm " + fnpat
                subprocess.os.system(command)
            comm.Barrier()

        comm.Barrier()

        isubvol_end = time()
        subvol_runtime_sec = isubvol_end - isubvol_start
        subvol_runtime_hours = subvol_runtime_sec / 3600
        if rank == 0:
            msg = f"\nRuntime for subvol {isubvol} = {subvol_runtime_hours:.4f} hours"
            print(msg)

    comm.Barrier()

    end = time()
    runtime_minutes = (end - start) / 60
    if rank == 0:
        print(f"Finished all {n_subvols} subvolumes, each with {nchunks} chunks")
        print(f"Total runtime for script = {runtime_minutes:.1f} minutes")
