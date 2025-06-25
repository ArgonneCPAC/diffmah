#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=4:mpiprocs=30

#PBS -l walltime=10:00:00

# Load software
source ~/.bash_profile
conda activate improv311

cd /home/ahearin/work/random/0625/LJ_DIFFMAH_JOBS
rsync /home/ahearin/work/repositories/python/diffmah/scripts/fitting_scripts/hacc_lastjourney_diffmah_fitter_script.py ./
mpirun -n 120 python hacc_lastjourney_diffmah_fitter_script.py /lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest /lcrc/project/halotools/LastJourney/diffmah_fits_littleh diffmah_fits.h5 -istart 0 -iend 20

rsync /home/ahearin/work/repositories/python/diffmah/scripts/fitting_scripts/unchunk_last_journey_script.py ./
python unchunk_last_journey_script.py 0 20 -indir /lcrc/project/halotools/LastJourney/diffmah_fits_littleh -outdir /lcrc/project/halotools/LastJourney/diffmah_fits_littleh/unchunked