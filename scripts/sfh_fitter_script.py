"""Script to fit Bolshoi or Multidark MAHs with a smooth model."""
import numpy as np
import os
from mpi4py import MPI
import argparse
from time import time
from load_um_sfh_data import load_mdpl2_data, load_bpl_data
from sfh_fitter_wrappers import get_mah_loss_args, _mah_loss
from sfh_fitter_wrappers import get_outline, get_header
from sfh_fitter_wrappers import get_sfh_loss_args, _mstar_loss
from sfh_fitter_wrappers import get_ssfr_loss_args, _ssfr_loss
from sfh_fitter_wrappers import _mstar_loss_wrapper, get_mstar_loss_wrapper_args

from diffmah.utils import jax_adam_wrapper
import subprocess
import h5py


def _write_collated_data(outname, data):
    nrows, ncols = np.shape(data)
    colnames = get_header()[1:].strip().split()
    msg = "Ncols in header = {0}\nNcols in data = {1}\nHeader = \n{2}\n"
    assert len(colnames) == ncols, msg.format(len(colnames), ncols, get_header())
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

    parser.add_argument("simulation", choices=("bpl", "mdpl"))
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output file")
    parser.add_argument(
        "loss", help="Name of the loss function", choices=("mstar", "ssfr", "mstar2")
    )
    parser.add_argument("-fstar_tau", help="Fstar timescale in Gyr", default=1.0)
    parser.add_argument("-indir", help="Input directory", default=None)
    parser.add_argument("-nstep", help="Num opt steps per halo", type=int, default=500)
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-ntest", help="Length of test run", type=int, default=4)
    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    nstep = args.nstep

    if args.simulation == "bpl":
        log_mah_min = 10.5
        if args.indir is None:
            all_halos, t_sim, z_sim = load_bpl_data()
        else:
            all_halos, t_sim, z_sim = load_bpl_data(args.indir)
    elif args.simulation == "mdpl":
        log_mah_min = 11.5
        if args.indir is None:
            all_halos, t_sim, z_sim = load_mdpl2_data()
        else:
            all_halos, t_sim, z_sim = load_mdpl2_data(args.indir)
    mah_kwargs = dict(log_mah_min=log_mah_min)

    if args.loss == "mstar":
        get_loss_args = get_sfh_loss_args
        loss_func = _mstar_loss

    elif args.loss == "ssfr":
        get_loss_args = get_ssfr_loss_args
        loss_func = _ssfr_loss
    elif args.loss == "mstar2":
        get_loss_args = get_mstar_loss_wrapper_args
        loss_func = _mstar_loss_wrapper

    TMP_OUTPAT = "_" + str(args.loss) + "_tmp_sfh_fits_rank_{0}.dat"
    rank_outname = os.path.join(args.outdir, TMP_OUTPAT).format(rank)

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * args.ntest
    else:
        nhalos_tot = len(all_halos)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]
    halos_for_rank = all_halos[indx]
    nhalos_for_rank = len(halos_for_rank)

    header = get_header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halos_for_rank["halo_id"][i]
            lgmah = halos_for_rank["log_mah"][i, :]
            tmp = halos_for_rank["tmp"][i]
            smh = halos_for_rank["smh"][i, :]
            log_ssfrh = halos_for_rank["log_ssfrh"][i, :]

            halo_id = halos_for_rank["halo_id"][i]
            log_mah_sim = halos_for_rank["log_mah"][i, :]
            sfh_sim = halos_for_rank["sfr_history_main_prog"][i, :]
            tmp_sim = halos_for_rank["tmp"][i]

            p_mah_init, mah_loss_data = get_mah_loss_args(
                t_sim, log_mah_sim, sfh_sim, tmp_sim, **mah_kwargs
            )

            mah_fit_data = jax_adam_wrapper(
                _mah_loss, p_mah_init, mah_loss_data, 200, n_warmup=1
            )
            mah_params = mah_fit_data[0]

            p_sfh_init, sfh_loss_data = get_loss_args(mah_params, mah_loss_data)
            sfh_fit_data = jax_adam_wrapper(
                loss_func, p_sfh_init, sfh_loss_data, 1000, n_warmup=2
            )

            outline = get_outline(
                halo_id, mah_fit_data, sfh_fit_data, mah_loss_data, sfh_loss_data
            )
            #
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
