"""
"""
from scipy.stats.mstats import trimmed_mean, trimmed_std
import numpy as np
import warnings


def get_clean_sample_mask(log_mah_fit, logmp_sample, it_min, lim=0.01, z_cut=3):
    n_h, n_t = log_mah_fit.shape
    log_mah_scaled = log_mah_fit - log_mah_fit[:, -1].reshape((-1, 1)) + logmp_sample
    clean_mask = np.ones(n_h).astype(bool)
    for it in range(it_min, n_t - 1):
        log_mah_at_t = log_mah_scaled[:, it]
        mu_t = trimmed_mean(log_mah_at_t, limits=(lim, lim))
        std_t = trimmed_std(log_mah_at_t, limits=(lim, lim))
        z_score_at_t = (log_mah_at_t - mu_t) / std_t
        clean_mask &= np.abs(z_score_at_t) < z_cut
    return clean_mask


def measure_target_data(mah, dmhdt, lgt, lgt_target, logmp_sample):
    mah0 = mah[:, -1].reshape(-1, 1)
    mp_sample = 10 ** logmp_sample
    scaled_mah = mp_sample * mah / mah0
    scaled_dmhdt = mp_sample * dmhdt / mah0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mean_mah_table = np.mean(scaled_mah, axis=0)
        mean_log_mah_table = np.mean(np.log10(scaled_mah), axis=0)
        std_log_mah_table = np.std(np.log10(scaled_mah), axis=0)

        mean_dmhdt_table = np.mean(scaled_dmhdt, axis=0)
        std_dmhdt_table = np.std(scaled_dmhdt, axis=0)

        mean_mah = 10 ** np.interp(lgt_target, lgt, np.log10(mean_mah_table))
        mean_log_mah = 10 ** np.interp(lgt_target, lgt, np.log10(mean_log_mah_table))
        mean_dmhdt = 10 ** np.interp(lgt_target, lgt, np.log10(mean_dmhdt_table))

        std_dmhdt = 10 ** np.interp(lgt_target, lgt, np.log10(std_dmhdt_table))
        std_log_mah = np.interp(lgt_target, lgt, std_log_mah_table)

    var_dmhdt = std_dmhdt ** 2
    var_log_mah = std_log_mah ** 2
    return mean_mah, mean_log_mah, var_log_mah, mean_dmhdt, var_dmhdt
