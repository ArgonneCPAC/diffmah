"""Script to fit Bolshoi, MDPL2, or TNG MAHs with the diffmah model."""

import argparse
import os
import subprocess
from glob import glob
from time import time

import numpy as np
from diffmah import fitting_helpers as cfh
from mpi4py import MPI

from load_w0wa_cores import NCHUNKS, NUM_SUBVOLS_DISCOVERY, load_mahs_per_rank

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"

LCRC = None
TASSO_ROOT_DRN = "/Users/aphearin/work/DATA/DESI_W0WA"
TASSO_LCDM_DRN = os.path.join(TASSO_ROOT_DRN, "LCDM")
TASSO_W0WA_DRN = os.path.join(TASSO_ROOT_DRN, "W0WA")

BNPAT = "m000p.coreforest.{}.hdf5"


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
        default=NUM_SUBVOLS_DISCOVERY,
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

    if args.indir == "TASSO":
        indir = TASSO_LCDM_DRN
        fn_cfg = glob(os.path.join(indir, "*.cfg"))[0]
    else:
        indir = args.indir
        # On LCRC the cfg file is one directory up from the data
        fn_cfg = glob(os.path.join(os.path.dirname(indir), "*.cfg"))[0]

    start = time()

    all_avail_subvol_fnames = glob(os.path.join(indir, "m000p.coreforest.*.hdf5"))
    all_avail_subvol_bnames = [os.path.basename(fn) for fn in all_avail_subvol_fnames]
    all_avail_subvolumes = [int(s.split(".")[2]) for s in all_avail_subvol_bnames]
    all_avail_subvolumes = np.array(sorted(all_avail_subvolumes))

    if args.test:
        subvolumes = [all_avail_subvolumes[0]]
        chunks = [0, 1, 2]
    else:
        subvolumes = all_avail_subvolumes[istart:iend]
        chunks = np.arange(nchunks).astype(int)

    for isubvol in subvolumes:
        isubvol_start = time()

        nchar_subvol = len(str(num_subvols_tot))

        subvol_str = f"{isubvol}"
        bname = BNPAT.format(subvol_str)
        fn_data = os.path.join(indir, bname)

        for chunknum in chunks:
            comm.Barrier()
            ichunk_start = time()

            tarr, mahs_for_rank = load_mahs_per_rank(
                fn_data, fn_cfg, chunknum, nchunks=nchunks, comm=MPI.COMM_WORLD
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
                print("\nFinished with subvolume {} chunk {}".format(
                    isubvol, chunknum
                ))
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
