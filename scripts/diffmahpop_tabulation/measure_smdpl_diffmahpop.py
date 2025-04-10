"""This script tabulates diffmah samples in SMDPL for diffmahpop
"""

import argparse
from time import time

import smdpl_diffmahpop_utils as smah_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("outdrn", help="output directory", type=str)
    parser.add_argument("-istart", help="start of subvolume loop", type=int, default=0)
    parser.add_argument(
        "-iend",
        help="end of subvolume loop",
        type=int,
        default=smah_utils.N_SUBVOL_SMDPL,
    )
    parser.add_argument(
        "-diffmah_drn",
        help="Directory storing diffmah fits",
        type=str,
        default=smah_utils.LCRC_DIFFMAH_DRN,
    )
    parser.add_argument(
        "-sfhcat_drn",
        help="Directory storing SFH binaries read by umachine_pyio",
        type=str,
        default=smah_utils.LCRC_SFHCAT_DRN,
    )
    parser.add_argument(
        "-n_m", help="Number of mass bins", type=int, default=smah_utils.N_LGM_BINS
    )
    parser.add_argument(
        "-n_t", help="Number of time bins", type=int, default=smah_utils.N_TIMES
    )

    args = parser.parse_args()
    outdrn = args.outdrn
    diffmah_drn = args.diffmah_drn
    sfhcat_drn = args.sfhcat_drn
    istart = args.istart
    iend = args.iend
    n_m = args.n_m
    n_t = args.n_t

    start = time()
    smah_utils.smdpl_diffmahpop_subvolume_loop(
        outdrn,
        diffmah_drn=diffmah_drn,
        sfhcat_drn=sfhcat_drn,
        istart=istart,
        iend=iend,
        n_m=n_m,
        n_t=n_t,
    )
    end = time()
    runtime = end - start
    print(f"Runtime = {runtime:.1f} seconds")
