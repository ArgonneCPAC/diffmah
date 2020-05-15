# diffmah

For a typical development environment in conda:

```bash
conda create -n diffit python=3.7 numpy numba emcee flake8 pytest tqdm pyyaml jax ipython jupyter matplotlib scipy h5py
```

You can then install the package into your environment with 
```
$ conda activate diffit
$ cd /path/to/root/diffmah
$ python setup.py install
```

Simulation data formatted for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/umachine_sfh_jonas/full_histories/binary_column_store).
