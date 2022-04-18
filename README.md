# diffmah

## Installation
For a typical development environment in conda:

```
$ conda config --add channels conda-forge
$ conda config --prepend channels conda-forge
$ conda create -n diffit python=3.7 numpy numba flake8 pytest jax ipython jupyter matplotlib scipy h5py
```

You can install the latest release of diffmah using conda or pip. To install diffmah into your environment from the source code:
```
$ conda activate diffit
$ cd /path/to/root/diffmah
$ python setup.py install
```

Data for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/diffmah_data/).

## Scripts and demo notebooks
The `diffmah_halo_populations.ipynb` notebook demonstrates how to calculate the MAHs as a function of the diffmah parameters using the `calc_halo_history` function. This notebook also demonstrates how to use the `mc_halo_population` function to generate Monte Carlo realizations of cosmologically representative populations of halos.

The `diffmah_fitter_demo.ipynb` notebook demonstrates how to fit the MAH of a simulated halo with a diffmah approximation.

See `history_fitting_script.py` for an example of how to fit the MAHs of a large number of simulated halos in parallel with mpi4py.

## Citing diffmah
[The diffmah paper](https://astro.theoj.org/article/26991-a-differentiable-model-of-the-assembly-of-individual-and-populations-of-dark-matter-halos) has been published by the [Open Journal of Astrophysics](https://astro.theoj.org/). Citation information for the paper can be found at [this ADS link](https://ui.adsabs.harvard.edu/abs/2021OJAp....4E...7H/abstract), copied below for convenience:

```
@ARTICLE{2021OJAp....4E...7H,
       author = {{Hearin}, Andrew P. and {Chaves-Montero}, Jon{\'a}s and {Becker}, Mathew R. and {Alarcon}, Alex},
        title = "{A Differentiable Model of the Assembly of Individual and Populations of Dark Matter Halos}",
      journal = {The Open Journal of Astrophysics},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2021,
        month = jul,
       volume = {4},
       number = {1},
          eid = {7},
        pages = {7},
          doi = {10.21105/astro.2105.05859},
archivePrefix = {arXiv},
       eprint = {2105.05859},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021OJAp....4E...7H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
