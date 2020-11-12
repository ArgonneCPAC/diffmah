"""Script to fit Bolshoi or Multidark MAHs with a smooth model."""
import numpy as np
import os
from mpi4py import MPI
import argparse
from time import time
from load_um_sfh_data import load_mdpl2_data, load_bpl_data
from fit_sfh import get_moster17_loss_data
from fit_sfh import moster17_loss
from fit_sfh import get_outline
from fit_sfh import DEFAULT_MAH_PARAMS, DEFAULT_SFR_PARAMS, DEFAULT_Q_PARAMS
from diffmah.utils import jax_adam_wrapper
from astropy.cosmology import Planck15
import subprocess
import h5py

TMP_OUTPAT = "_tmp_sfh_fits_rank_{0}.dat"

T_MAH_MIN = 1.0
T_SMH_MIN = 1.0
T_SSFRH_MIN = 1.0
DMHDT_K = DEFAULT_MAH_PARAMS["dmhdt_k"]
LOG_SSFR_CLIP = -11.0
DLOGMH_CUT = 3.0
DLOGSM_CUT = 2.0

Z_TABLE = np.linspace(30, 0.0, 1000)
T_TABLE = Planck15.age(Z_TABLE).value


def _write_collated_data(outname, data):
    nrows, ncols = np.shape(data)
    colnames = _get_header()[1:].strip().split()
    msg = "Ncols in header = {0}\nNcols in data = {1}\nHeader = \n{2}\n"
    assert len(colnames) == ncols, msg.format(len(colnames), ncols, _get_header())
    with h5py.File(outname, "w") as hdf:
        for i, name in enumerate(colnames):
            if name == "halo_id":
                hdf[name] = data[:, i].astype("i8")
            else:
                hdf[name] = data[:, i]


def _get_header():
    mah_keys = tuple(DEFAULT_MAH_PARAMS.keys())
    sfr_keys = tuple(DEFAULT_SFR_PARAMS.keys())
    q_keys = tuple(DEFAULT_Q_PARAMS.keys())
    final_keys = ("tmp", "loss")
    colnames = ("#", "halo_id", "logmp", *mah_keys, *sfr_keys, *q_keys, *final_keys)
    header = " ".join(colnames) + "\n"
    return header


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("simulation", choices=("bpl", "mdpl"))
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output file")
    parser.add_argument("-indir", help="Input directory", default=None)
    parser.add_argument("-nstep", help="Num opt steps per halo", type=int, default=500)
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    rank_outname = os.path.join(args.outdir, TMP_OUTPAT).format(rank)
    nstep = args.nstep

    if args.simulation == "bpl":
        if args.indir is None:
            all_halos, t_sim, z_sim = load_bpl_data()
        else:
            all_halos, t_sim, z_sim = load_bpl_data(args.indir)
    elif args.simulation == "mdpl":
        if args.indir is None:
            all_halos, t_sim, z_sim = load_mdpl2_data()
        else:
            all_halos, t_sim, z_sim = load_mdpl2_data(args.indir)

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 4
    else:
        nhalos_tot = len(all_halos)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]
    halos_for_rank = all_halos[indx]
    nhalos_for_rank = len(halos_for_rank)

    header = _get_header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halos_for_rank["halo_id"][i]
            lgmah = halos_for_rank["log_mah"][i, :]
            tmp = halos_for_rank["tmp"][i]
            smh = halos_for_rank["smh"][i, :]
            log_ssfrh = halos_for_rank["log_ssfrh"][i, :]

            p_init, loss_data = get_moster17_loss_data(
                T_TABLE,
                Z_TABLE,
                tmp,
                t_sim,
                lgmah,
                smh,
                log_ssfrh,
                T_MAH_MIN,
                T_SMH_MIN,
                T_SSFRH_MIN,
                DMHDT_K,
                LOG_SSFR_CLIP,
                DLOGMH_CUT,
                DLOGSM_CUT,
            )
            fit_data = jax_adam_wrapper(
                moster17_loss, p_init, loss_data, nstep, n_warmup=1
            )
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
