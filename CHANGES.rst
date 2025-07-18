0.7.3 (2025-07-17)
------------------
- Update fitter scripts for HACC simulations


0.7.2 (2025-05-19)
------------------
- Include optional keyword argument `tpeak_fixed` to `diffmah_fitter` (https://github.com/ArgonneCPAC/diffmah/pull/168)
- Add new convenience kernel `logmh_at_t_obs` for calculating masses of halos on a lightcone (https://github.com/ArgonneCPAC/diffmah/pull/171)


0.7.1 (2025-03-24)
------------------
- Include convenience function mc_select to choose MC realization of MAH (https://github.com/ArgonneCPAC/diffmah/pull/165)


0.7.0 (2025-01-13)
------------------
- Remove old and obsolete code (https://github.com/ArgonneCPAC/diffmah/pull/159)


0.6.3 (2024-12-14)
------------------
- Require python>=3.11 (https://github.com/ArgonneCPAC/diffmah/pull/158)


0.6.2 (2024-12-14)
------------------
- Add data_loader sub-package (https://github.com/ArgonneCPAC/diffmah/pull/153)

- Add script for fitting HACC core merger trees (https://github.com/ArgonneCPAC/diffmah/pull/152)

- Change t_peak param bounds (https://github.com/ArgonneCPAC/diffmah/pull/156)

- Change API of DiffmahPop MC generators (https://github.com/ArgonneCPAC/diffmah/pull/155)


0.6.1 (2024-10-24)
------------------
- Include t_peak as fifth diffmah parameter (https://github.com/ArgonneCPAC/diffmah/pull/147)


0.6.0 (2024-08-09)
------------------
- Add prototype for P(MAH | m_obs, t_obs) (https://github.com/ArgonneCPAC/diffmah/pull/132)


0.5.1 (2024-08-07)
------------------
- Add new convenience kernels for diffstar (https://github.com/ArgonneCPAC/diffmah/pull/131)


0.5.0 (2024-01-15)
------------------
- namedtuple DEFAULT_MAH_PARAMS now imported from diffmah.defaults (https://github.com/ArgonneCPAC/diffmah/pull/113)


0.4.3 (2024-01-15)
------------------
- Change DEFAULT_MAH_PARAMS to namedtuple instead of ndarray (https://github.com/ArgonneCPAC/diffmah/pull/112)


0.4.2 (2023-09-26)
------------------
- Remove optional mah_type argument of mc_halo_population function
- Implement differentiable kernel for mc_halo_population function
- Update requirement to python>=3.9
- Update packaging structure to pyproject.toml
- Remove diffmah/_version.py and switch to dynamic versioning


0.4.1 (2022-09-17)
------------------
- Fix bug in Monte Carlo generator of MAHs for halos identified at high redshift


0.4.0 (2022-08-27)
------------------
- Include prototype for Monte Carlo realizations of populations of halos identified at higher redshift


0.3.0 (2022-09-14)
------------------
- Include new kernel for MAH of an individual halo at a single scalar-valued time


0.2.0 (2022-04-18)
------------------
- Updated API of Monte Carlo generator function `mc_halo_population` to accept input array of different masses (https://github.com/ArgonneCPAC/diffmah/pull/90).
