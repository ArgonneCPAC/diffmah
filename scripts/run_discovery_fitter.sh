#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A cosmo_ai

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:ncpus=100:mpiprocs=100

#PBS -l walltime=06:00:00

# Load software
source ~/.bash_profile
conda activate improv311

cd /home/ahearin/work/random/1214
mpirun -n 100 python hacc_discovery_sims_diffmah_fitter_script.py LCRC W0WA /lcrc/project/halotools/DESI_W0WA/diffmah_fits/W0WA -istart 0 -iend 1
