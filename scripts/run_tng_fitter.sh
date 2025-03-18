#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:ncpus=100:mpiprocs=100

#PBS -l walltime=06:00:00

# Load software
source ~/.bash_profile
cd ~/source/diffmah/scripts/

mpirun -n 100 python tng_diffmah_fitter_script.py /lcrc/project/halotools/alarcon/data/ /lcrc/project/halotools/alarcon/results/tng_diffmah_tpeak/
