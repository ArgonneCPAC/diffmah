#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A halotools

# # queue name (compute is the default)
# #PBS -q compute

# allocate 1 nodes, each with 30 MPI processes
#PBS -l select=1:mpiprocs=30

#PBS -l walltime=01:00:00

# uncomment to pass in full environment
# #PBS -V

# Load software
source ~/.bash_profile
conda activate diffhacc

cd /home/ahearin/work/random/0925

rm -rf /lcrc/project/halotools/random_data/0925
mkdir /lcrc/project/halotools/random_data/0925
python measure_smdpl_diffmahpop.py /lcrc/project/halotools/random_data/0925
