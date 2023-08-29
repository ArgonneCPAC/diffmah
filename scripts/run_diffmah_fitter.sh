#!/bin/bash
#SBATCH -J mah200
#SBATCH -A halotools
#SBATCH -p bdw
#SBATCH --ntasks-per-node=36
#SBATCH -N 24
#SBATCH -t 24:00:00

source activate diffsky_env
srun -n 720 --cpu-bind=cores python smdpl_fitting_script.py BEBOP /lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/diffmah_fits diffmah_fits.h5 -istart 200
