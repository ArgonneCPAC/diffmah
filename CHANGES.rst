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
