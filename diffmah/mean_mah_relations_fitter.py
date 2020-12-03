"""
"""
import numpy as np


def measure_target_data(
    all_log_mahs,
    all_dmhdts,
    t_sim,
    t_target,
    logmp_sample,
    t_min,
    log_mah_min,
    counts_min=20,
):
    all_logmp = all_log_mahs[:, -1]
    logmp_msk = np.abs(all_logmp - logmp_sample) < 0.1

    _log_mahs = all_log_mahs[logmp_msk]
    _dmhdts = all_dmhdts[logmp_msk]

    _msk_lgmin = _log_mahs > log_mah_min
    _c = np.sum(_msk_lgmin, axis=0)
    msk_c = _c > counts_min
    msk_t = t_sim >= t_min
    _a = np.arange(len(t_sim)).astype("i4")
    ifirst = _a[msk_c & msk_t][0]
    log_mahs = _log_mahs[:, ifirst:]
    dmhdts = _dmhdts[:, ifirst:]
    t_table = t_sim[ifirst:]

    msk_lgmin = log_mahs > log_mah_min

    ws = np.sum(msk_lgmin * 10 ** log_mahs, axis=0)
    ws2 = np.sum(msk_lgmin * dmhdts, axis=0)
    counts = np.sum(msk_lgmin, axis=0)
    mean_mah_table = ws / counts
    mean_dmhdt_table = ws2 / counts

    d_dhmdt_sq = (dmhdts - mean_dmhdt_table.reshape((1, -1))) ** 2
    ws3 = np.sum(msk_lgmin * d_dhmdt_sq, axis=0)
    var_dmhdt = ws3 / counts
    std_dmhdt_table = np.sqrt(var_dmhdt)

    mean_mah_target = 10 ** np.interp(
        np.log10(t_target), np.log10(t_table), np.log10(mean_mah_table)
    )
    mean_dmhdt_target = 10 ** np.interp(
        np.log10(t_target), np.log10(t_table), np.log10(mean_dmhdt_table)
    )

    std_dmhdt_target = 10 ** np.interp(
        np.log10(t_target), np.log10(t_table), np.log10(std_dmhdt_table)
    )

    return mean_mah_target, mean_dmhdt_target, std_dmhdt_target
