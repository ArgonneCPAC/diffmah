#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=4:mpiprocs=30

#PBS -l walltime=4:00:00

# Load software
source ~/.bash_profile
cd /home/ahearin/work/random/0410

mpiexec -n 120 python galacticus_diffmah_fitter_script.py