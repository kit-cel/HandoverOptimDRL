"""Utility functions for the project."""

import re
import numpy as np


def extract_speed(filename: str) -> int:
    """
    Extract speed from filename.

    Parameters
    ----------
    filename : str
        Filename string.

    Returns
    -------
    int
        Speed in km/h.
    """
    match = re.search(r"(\d+)kmh", filename)
    return int(match.group(1)) if match else None


def filenames_speed_filter(
    rsrp_filenames: list[str],
    sinr_filenames: list[str],
    use_speed_list: list[int],
) -> tuple[list[str], list[str], list[int]]:
    """
    Filter the files based on the speed list.

    Parameters
    ----------
    rsrp_filenames : list[str]
        List of RSRP filenames.
    sinr_filenames : list[str]
        List of SINR filenames.
    use_speed_list : list[int]
        List of speeds to use.

    Returns
    -------
    tuple[list[str], list[str], list[int]]
        Filtered RSRP filenames, SINR filenames, and speeds
    """
    # Extract speed from filename "ue999kmh_...mat"
    speeds = [extract_speed(f) for f in rsrp_filenames]

    # Filter the dataset based on the speed
    idxs = [i for i, speed in enumerate(speeds) if speed in use_speed_list]
    rsrp_filenames = [rsrp_filenames[i] for i in idxs]
    sinr_filenames = [sinr_filenames[i] for i in idxs]
    speeds = [speeds[i] for i in idxs]
    return rsrp_filenames, sinr_filenames, speeds


def get_sync_state(sinr_db: float, q_in: float, q_out: float):
    """
    Check if the SINR is within the QoS thresholds.

    Parameters
    ----------
    sinr_db : float
        SINR in dB.
    q_in : float
        Upper QoS threshold in dB. Indicates in-sync.
    q_out : float
        Lower QoS threshold in dB. Indicates out-of-sync.

    Returns
    -------
    int
        State of the SINR. 1: out-of-sync, 2: in hysteresis, 3: in-sync.
    """
    if sinr_db < q_out:
        return 1
    if sinr_db < q_in:
        return 2
    return 3


def norm(ins: np.ndarray, clip_l: float, clip_h: float) -> np.ndarray:
    """
    Clip and normalize the input values.

    Parameters
    ----------
    ins : np.ndarray
        Input values.
    clip_l : float
        Lower clipping value.
    clip_h : float
        Upper clipping value.

    Returns
    -------
    np.ndarray
        Normalized values between 0 and 1.
    """
    outs = np.clip(ins, clip_l, clip_h)
    return (outs - clip_l) / (clip_h - clip_l)


def mwatt_to_dbm(mwatt: float | np.ndarray) -> float | np.ndarray:
    """
    Convert mW to dBm.

    Parameters
    ----------
    mwatt : float | np.ndarray
        Power in mW.

    Returns
    -------
    float | np.ndarray
        Power in dBm.
    """
    return 10 * np.log10(mwatt)


def dbm_to_mwatt(dbm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert dBm to mW.

    Parameters
    ----------
    dbm : float | np.ndarray
        Power in dBm.

    Returns
    -------
    float | np.ndarray
        Power in mW.
    """
    return np.power(10, dbm / 10)


def get_result_container(speeds: list[int]) -> dict:
    """
    Get the result container for additional statistics.

    Parameters
    ----------
    speeds : list[int]
        List of speeds.

    Returns
    -------
    dict
        Result container.
    """
    sinr_connected = {}
    sinr_max = {}
    n_ho = {}
    n_pp = {}
    n_rlf = {}
    sinr_at_ho_exe_pcell = []
    sinr_after_ho_exe_tcell = []
    for speed in speeds:
        sinr_connected[speed] = []
        sinr_max[speed] = []
        n_ho[speed] = []
        n_pp[speed] = []
        n_rlf[speed] = []

    res_dict = {
        "sinr_connected": sinr_connected,
        "sinr_max": sinr_max,
        "n_ho": n_ho,
        "n_pp": n_pp,
        "n_rlf": n_rlf,
        "sinr_at_ho_exe_pcell": sinr_at_ho_exe_pcell,
        "sinr_after_ho_exe_tcell": sinr_after_ho_exe_tcell,
    }
    return res_dict


def print_statistics(stat_dict: dict[str, int | float]) -> None:
    """
    Print statistics.

    Parameters
    ----------
    stat_dict : dict[str, int | float]
        Dictionary containing the statistics.
    """
    params = [
        "Parameter",
        "Num BS",
        "Num samples",
        "",
        "Mean SINR",
        "Variance SINR",
        "Median SINR",
        "Q1 SINR",
        "Q3 SINR",
        "",
        "Max capacity",
        "Achieved capacity",
        "Total time",
        "Connected time",
        "",
        "# HO prep started",
        "# HO prep terminated",
        "# HO prep completed",
        "# HO prep aborted",
        "# HO prep failed",
        "# HO prep error",
        "",
        "# HO exe started",
        "# HO exe terminated",
        "# HO exe completed",
        "# HO exe aborted",
        "# HO exe failed",
        "# HO exe error",
        "",
        "# RLF",
        "# RLF rate",
        "# PP",
        "# PP rate",
    ]
    values = [
        "Value",
        f"{stat_dict['num_base_stations']}",
        f"{stat_dict['num_samples']}",
        "",
        f"{stat_dict['mean_sinr']:.3f} dB",
        f"{stat_dict['variance_sinr']:.3f} dB^2",
        f"{stat_dict['median_sinr']:.3f} dB",
        f"{stat_dict['q1_sinr']:.3f} dB",
        f"{stat_dict['q3_sinr']:.3f} dB",
        "",
        f"{stat_dict['max_spectral_eff']:.3f} bits/s/Hz",
        f"{stat_dict['spectral_eff']:.3f} bits/s/Hz |"
        + f" {100 * stat_dict['spectral_eff'] / stat_dict['max_spectral_eff']:.3f}%",
        f"{stat_dict['total_time']} samples",
        f"{stat_dict['connected_time']} samples |"
        + f" {100 * stat_dict['connected_time'] / stat_dict['total_time']:.3f}%",
        "",
        f"{stat_dict['num_ho_prep_started']}",
        f"{stat_dict['num_ho_prep_terminated']}",
        f"{stat_dict['num_ho_prep_completed']}",
        f"{stat_dict['num_ho_prep_aborted']}",
        f"{stat_dict['num_ho_prep_failed']}",
        f"{stat_dict['num_ho_prep_error']}",
        "",
        f"{stat_dict['num_ho_exe_started']}",
        f"{stat_dict['num_ho_exe_terminated']}",
        f"{stat_dict['num_ho_exe_completed']}",
        f"{stat_dict['num_ho_exe_aborted']}",
        f"{stat_dict['num_ho_exe_failed']}",
        f"{stat_dict['num_ho_exe_error']}",
        "",
        f"{stat_dict['num_rlf']}",
        f"{100 * stat_dict['rlf_rate']:.3f}%",
        f"{stat_dict['num_pp']}",
        f"{100 * stat_dict['pp_rate']:.3f}%",
    ]
    tab = list(zip(params, values))
    max_param_len = max(len(row[0]) for row in tab)

    print("\nResults\n---------------------------------")
    for row in tab:
        print(f"{row[0]:<{max_param_len}}: {row[1]}")


def print_aggregated_stats(stats: dict):
    """
    Print statistics.

    Parameters
    ----------
    stats : dict
        Dictionary containing the aggregated statistics.
    """
    params = [
        "PARAMETERS",
        "Num datasets",
        "",
        "Avg relative rate (total)",
        "",
        "Speeds",
        "Avg relative rate",
        "Avg PP prob",
        "Avg RLF prob",
        "",
        "Avg # HO prep started",
        "Avg # HO prep terminated",
        "Avg # HO prep completed",
        "Avg # HO prep aborted",
        "Avg # HO prep failed",
        "Avg # HO prep error",
        "",
        "Avg # HO exe started",
        "Avg # HO exe terminated",
        "Avg # HO exe completed",
        "Avg # HO exe aborted",
        "Avg # HO exe failed",
        "Avg # HO exe error",
    ]
    values = [
        "VALUES",
        f"{len(stats['spectral_eff'])}",
        "",
        f"{np.mean(stats["r_rel"]):6.3f} %",
        "",
        (
            f"[{"".join([f"{speed:6d}, " for speed in stats['speeds']])}] km/h"
            if len(stats["speeds"]) > 1
            else f"{stats["speeds"][0]:6d} km/h"
        ),
        (
            f"[{"".join([f"{100*r_rel:6.3f}, " for r_rel in stats['r_rel']])}] %"
            if len(stats["r_rel"]) > 1
            else f"{100*stats['r_rel'][0]:6.3f} %"
        ),
        (
            f"[{"".join([f"{100*pp_prob:6.3f}, " for pp_prob in stats['mean_pp_prob']])}] %"
            if len(stats["mean_pp_prob"]) > 1
            else f"{100*stats['mean_pp_prob'][0]:6.3f} %"
        ),
        (
            f"[{"".join([f"{100*rlf_prob:6.3f}, " for rlf_prob in stats['mean_rlf_prob']])}] %"
            if len(stats["mean_rlf_prob"]) > 1
            else f"{100*stats['mean_rlf_prob'][0]:6.3f} %"
        ),
        "",
        f"{np.mean(stats['num_ho_prep_started']):5.2f}",
        f"{np.mean(stats['num_ho_prep_terminated']):5.2f}",
        f"{np.mean(stats['num_ho_prep_completed']):5.2f}",
        f"{np.mean(stats['num_ho_prep_aborted']):5.2f}",
        f"{np.mean(stats['num_ho_prep_failed']):5.2f}",
        f"{np.mean(stats['num_ho_prep_error']):5.2f}",
        "",
        f"{np.mean(stats['num_ho_exe_started']):5.2f}",
        f"{np.mean(stats['num_ho_exe_terminated']):5.2f}",
        f"{np.mean(stats['num_ho_exe_completed']):5.2f}",
        f"{np.mean(stats['num_ho_exe_aborted']):5.2f}",
        f"{np.mean(stats['num_ho_exe_failed']):5.2f}",
        f"{np.mean(stats['num_ho_exe_error']):5.2f}",
    ]
    tab = list(zip(params, values))
    max_param_len = max(len(row[0]) for row in tab)

    print("\nAggregated Results\n---------------------------------")
    for row in tab:
        print(f"{row[0]:<{max_param_len}}: {row[1]}")
