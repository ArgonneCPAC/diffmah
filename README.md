# diffmah

For a typical development environment in conda:

```bash
conda create -n diffit python=3.7 numpy numba flake8 pytest jax ipython jupyter matplotlib scipy h5py
```

You can install the package into your environment with
```
$ conda activate diffit
$ cd /path/to/root/diffmah
$ python setup.py install
```

Data for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/diffmah_data/).

## Demo notebooks
The `diffmah_halo_populations.ipynb` notebook demonstrates how to calculate the MAHs as a function of the diffmah parameters using the `calc_halo_history` function. This notebook also demonstrates how to use the `mc_halo_population` function to generate Monte Carlo realizations of cosmologically representative populations of halos.

The `diffmah_halo_populations.ipynb` notebook demonstrates how to fit the MAH of a simulated halo with a diffmah approximation.
