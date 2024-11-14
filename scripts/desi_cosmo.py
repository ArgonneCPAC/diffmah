"""On 11/11/2024 these numbers were copied and pasted from
https://docs.google.com/document/d/1VOyXhI6NbaNS-VeFIKnhdAA8GovEnVVoOC-0RpeAZwM/edit?pli=1&tab=t.0

"""

from dsps.cosmology import flat_wcdm

N_ptcl_per_dim = 6528

# LCDM
dlc_omega_cdm = 0.2589
dlc_omega_m = 0.3069
dlc_littleh = 0.6797
dlc_sigma8 = 0.8102
dlc_n_s = 0.9665
dlc_w_0 = -1.0
dlc_w_a = 0.0

dlc_Lbox = 1019.55  # = 1500 Mpc
dlc_mptcl = 3.26e8  # Msun/h

dlc_params = flat_wcdm.CosmoParams(dlc_omega_m, dlc_w_0, dlc_w_a, dlc_littleh)
t0_lcdm = flat_wcdm.age_at_z0(*dlc_params)

# w0-wa

dwc_omega_cdm = 0.296
dwc_omega_m = 0.344
dwc_littleh = 0.647
dwc_sigma8 = 0.8102
dwc_n_s = 0.9665
dwc_w_0 = -0.45
dwc_w_a = -1.79

dwc_Lbox = 970.5  # = 1500 Mpc
dwc_m_ptcl = 3.15e8  # Msun/h

dwc_params = flat_wcdm.CosmoParams(dwc_omega_m, dwc_w_0, dwc_w_a, dwc_littleh)
t0_w0wa = flat_wcdm.age_at_z0(*dwc_params)
