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
