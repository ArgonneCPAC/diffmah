# diffmah

## Installation
The latest release of diffmah is available for installation with pip or conda-forge:
```
$ conda install -c conda-forge diffmah
```

To install diffmah into your environment from the source code:

```
$ cd /path/to/root/diffmah
$ pip install .
```

### Environment configuration
For a typical development environment in conda-forge:

```
$ conda create -c conda-forge -n diffit python=3.9 numpy jax pytest ipython jupyter matplotlib scipy h5py diffmah
```

## Project data
Data for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/diffmah_data/).

## Documentation
Online documentation for Diffmah is available [diffmah.readthedocs.io](https://diffmah.readthedocs.io/en/latest/).

## Scripts and demo notebooks
The `diffmah_fitter_demo.ipynb` notebook demonstrates how to fit the MAH of a simulated halo with a diffmah approximation. See `history_fitting_script.py` for an example of how to fit the MAHs of a large number of simulated halos in parallel with mpi4py.

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
