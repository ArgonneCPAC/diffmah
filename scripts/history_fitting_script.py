"""Script to fit Bolshoi or Multidark MAHs with a smooth model."""
import numpy as np
import os
from mpi4py import MPI
import argparse
from time import time
from load_mah_data import (
    load_tng_data,
    TASSO,
    BEBOP,
    load_bolshoi_data,
    load_mdpl2_data,
)

from diffmah.fit_mah_helpers import _get_header, get_outline_bad_fit
from diffmah.fit_mah_helpers import get_loss_data_variable_mp_x0
from diffmah.fit_mah_helpers import get_loss_data_variable_mp, get_loss_data_fixed_mp
from diffmah.fit_mah_helpers import mse_loss_variable_mp_x0, mse_loss_fixed_mp
from diffmah.fit_mah_helpers import mse_loss_variable_mp
from diffmah.fit_mah_helpers import get_outline_variable_mp_x0, get_outline_fixed_mp
from diffmah.fit_mah_helpers import get_outline_variable_mp

from diffmah.fit_mah_helpers import lge_mse_loss_fixed_x0, get_loss_data_lge_fixed_x0
from diffmah.fit_mah_helpers import get_outline_lge_fixed_x0

from diffmah.fit_mah_helpers import lge_lgl_x0_mse_loss, get_loss_data_lge_lgl_x0
from diffmah.fit_mah_helpers import get_outline_lge_lgl_x0

from diffmah.utils import jax_adam_wrapper
import subprocess
import h5py

TMP_OUTPAT = "_tmp_mah_fits_rank_{0}.dat"
TODAY = 13.8


def _write_collated_data(outname, data):
    nrows, ncols = np.shape(data)
    colnames = _get_header()[1:].strip().split()
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

    parser.add_argument(
        "simulation", help="name of the simulation (used to select the data loader)"
    )
    parser.add_argument(
        "fit_type",
        help="Defines the loss function",
        choices=(
            "variable_mp_x0",
            "fixed_mp",
            "variable_mp",
            "lge_fixed_mp",
            "lge_lgl_x0",
        ),
    )
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output hdf5 file")
    parser.add_argument("-indir", help="Input directory", default="TASSO")
    parser.add_argument("-nstep", help="Num opt steps per halo", type=int, default=300)
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument(
        "-gal_type",
        help="Galaxy type (only relevant for Bolshoi and MDPl2)",
        default="cens",
    )

    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    rank_basepat = args.outbase + TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)
    nstep = args.nstep

    fit_type = args.fit_type
    if fit_type == "variable_mp_x0":
        mse_loss = mse_loss_variable_mp_x0
        get_loss_data = get_loss_data_variable_mp_x0
        get_outline = get_outline_variable_mp_x0
    elif fit_type == "fixed_mp":
        mse_loss = mse_loss_fixed_mp
        get_loss_data = get_loss_data_fixed_mp
        get_outline = get_outline_fixed_mp
    elif fit_type == "variable_mp":
        mse_loss = mse_loss_variable_mp
        get_loss_data = get_loss_data_variable_mp
        get_outline = get_outline_variable_mp
    elif fit_type == "lge_fixed_mp":
        mse_loss = lge_mse_loss_fixed_x0
        get_loss_data = get_loss_data_lge_fixed_x0
        get_outline = get_outline_lge_fixed_x0
    elif fit_type == "lge_lgl_x0":
        mse_loss = lge_lgl_x0_mse_loss
        get_loss_data = get_loss_data_lge_lgl_x0
        get_outline = get_outline_lge_lgl_x0
    else:
        raise ValueError("fit_type = {0} not recognized".format(fit_type))

    if args.indir == "TASSO":
        indir = TASSO
    elif args.indir == "BEBOP":
        indir = BEBOP
    else:
        indir = args.indir

    if args.simulation == "tng":
        _mah_data = load_tng_data(data_drn=indir)
        halo_ids, log_mahs, tmpeaks, tarr, lgm_min = _mah_data
    elif args.simulation == "bpl":
        _mah_data = load_bolshoi_data(args.gal_type, data_drn=indir)
        halo_ids, log_mahs, tmpeaks, tarr, lgm_min = _mah_data
    elif args.simulation == "mdpl":
        _mah_data = load_mdpl2_data(args.gal_type, data_drn=indir)
        halo_ids, log_mahs, tmpeaks, tarr, lgm_min = _mah_data
    else:
        raise NotImplementedError("Your data loader here")

    # Ensure the target MAHs are cumulative peak masses
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 5
    else:
        nhalos_tot = len(halo_ids)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]

    halo_ids_for_rank = halo_ids[indx]
    log_mahs_for_rank = log_mahs[indx]
    tmpeaks_for_rank = tmpeaks[indx]
    nhalos_for_rank = len(halo_ids_for_rank)

    header = _get_header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            lgmah = log_mahs_for_rank[i, :]
            tmp_fit = TODAY

            p_init, loss_data = get_loss_data(
                tarr,
                lgmah,
                tmp_fit,
                lgm_min,
            )
            _res = jax_adam_wrapper(mse_loss, p_init, loss_data, nstep, n_warmup=2)
            p_best, loss_best, loss_arr, params_arr, fit_terminates = _res

            if fit_terminates == 1:
                outline = get_outline(halo_id, loss_data, p_best, loss_best)
            else:
                outline = get_outline_bad_fit(halo_id, lgmah[-1], tmp_fit)

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
