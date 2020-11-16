"""Script to fit Bolshoi or Multidark MAHs with a smooth model."""
import numpy as np
import os
from mpi4py import MPI
import argparse
from time import time
from load_um_sfh_data import load_mdpl2_data, load_bpl_data
from fit_mah import get_loss_data_fixed_logmp, get_loss_data_fixed_k_x0
from fit_mah import mse_loss_fixed_logmp, mse_loss_fixed_k_x0
from fit_mah import get_outline_fixed_logmp, get_outline_fixed_k_x0
from diffmah.utils import jax_adam_wrapper
import subprocess
import h5py

TMP_OUTPAT = "_tmp_mah_fits_rank_{0}.dat"


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


def _get_header():
    return (
        "# halo_id logmp dmhdt_x0 dmhdt_k dmhdt_early_index dmhdt_late_index tmp loss\n"
    )


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("simulation", choices=("bpl", "mdpl"))
    parser.add_argument("indir", help="Input directory")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output file")
    parser.add_argument(
        "modelname",
        help="Version of the model and loss",
        choices=("fixed_logmp", "fixed_k_x0"),
    )
    parser.add_argument("-nstep", help="Num opt steps per halo", type=int, default=200)
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    rank_outname = os.path.join(args.outdir, TMP_OUTPAT).format(rank)
    nstep = args.nstep

    if args.simulation == "bpl":
        all_halos, tarr, zarr = load_bpl_data(args.indir)
    elif args.simulation == "mdpl":
        all_halos, tarr, zarr = load_mdpl2_data(args.indir)

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 2
    else:
        nhalos_tot = len(all_halos)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]
    halos_for_rank = all_halos[indx]
    nhalos_for_rank = len(halos_for_rank)

    # Define the loss function, which params are varied, etc.
    if args.modelname == "fixed_logmp":
        get_loss_data = get_loss_data_fixed_logmp
        loss_func = mse_loss_fixed_logmp
        get_outline = get_outline_fixed_logmp
    elif args.modelname == "fixed_k_x0":
        get_loss_data = get_loss_data_fixed_k_x0
        loss_func = mse_loss_fixed_k_x0
        get_outline = get_outline_fixed_k_x0

    header = _get_header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halos_for_rank["halo_id"][i]
            lgmah = halos_for_rank["log_mah"][i, :]
            lgdmhdt = halos_for_rank["log_dmhdt"][i, :]
            tmp = halos_for_rank["tmp"][i]

            loss_data, p_init = get_loss_data(tarr, lgmah, lgdmhdt, tmp)
            fit_data = jax_adam_wrapper(loss_func, p_init, loss_data, nstep)
            outline = get_outline(halo_id, tmp, loss_data, fit_data)

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
        pat = os.path.join(args.outdir, TMP_OUTPAT)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        data_collection = [np.loadtxt(fn) for fn in fit_data_fnames]
        all_fit_data = np.concatenate(data_collection)
        outname = os.path.join(args.outdir, args.outbase)
        _write_collated_data(outname, all_fit_data)

        #  clean up temporary files
        _remove_basename = pat.replace("{0}", "*")
        command = "rm -rf " + _remove_basename
        raw_result = subprocess.check_output(command, shell=True)
