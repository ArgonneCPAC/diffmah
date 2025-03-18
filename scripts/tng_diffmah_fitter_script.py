"""Script to fit Bolshoi, MDPL2, or TNG MAHs with the diffmah model."""

import argparse
import os
import subprocess
from glob import glob
from time import time

import numpy as np
from mpi4py import MPI

from diffmah.fitting_helpers import diffmah_fitter_helpers as cfh
from diffmah.data_loaders.load_tng_mahs import load_tng_data

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"

LCRC_DRN_PAT = "/lcrc/project/halotools/alarcon/data/"

NCHUNKS = 20

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Root directory storing TNG sims")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument(
        "-outbase", help="Basename of the output hdf5 file", default="diffmah_fits.h5"
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-nstep", help="Number of gd steps", type=int, default=200)
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)

    args = parser.parse_args()
    nstep = args.nstep
    indir = args.indir
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    nchar_chunks = len(str(nchunks))

    os.makedirs(outdir, exist_ok=True)

    start = time()

    tarr, mahs = load_tng_data(data_drn=indir)

    if args.test:
        chunks = [0, 1, 2]
    else:
        chunks = np.arange(nchunks).astype(int)

    nhalos_tot = len(mahs)

    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]

    mahs_for_rank = mahs[indx]
    nhalos_for_rank = len(mahs_for_rank)

    comm.Barrier()
    ichunk_start = time()

    nhalos_for_rank = mahs_for_rank.shape[0]
    nhalos_tot = comm.reduce(nhalos_for_rank, op=MPI.SUM)

    chunknum = rank

    rank_basepat = TMP_OUTPAT
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
        outbn = "diffmah_tng_fits.hdf5"
        outfn = os.path.join(outdir, outbn)

        cfh.write_collated_data(outfn, chunk_fit_results)

        # clean up ASCII data for subvol_i
        bn = fit_data_bnames[0]
        bnpat = "_".join(bn.split("_")[:-1]) + "_*.dat"
        fnpat = os.path.join(outdir, bnpat)
        command = "rm " + fnpat
        subprocess.os.system(command)
