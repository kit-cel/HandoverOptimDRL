"""Validate 3GPP protocol on the handover environment."""

import os
import sys

import numpy as np

from src.config import Config
import src.dataloader as dl
from src.env import HandoverEnv3GPP
import src.utils as ut


THIS_PATH = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    """Main function."""

    config = Config()
    # Load MATLAB files
    data_dir = os.path.join(THIS_PATH, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")

    # Speed filter
    use_speed_list = [30, 50, 70, 90]
    rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(
        rsrp_files, sinr_files, use_speed_list
    )

    # Result containers
    env = HandoverEnv3GPP(config)
    aggregated_stats = {key: [] for key in env.get_stats_dict()}
    result_container = ut.get_result_container(speeds)

    # Run the 3GPP handover protocol
    for i, (speed, rsrp_filename, sinr_filename) in enumerate(
        zip(speeds, rsrp_files, sinr_files)
    ):
        print(f"Testing 3GPP HO protocol with dataset {i:3d}/{len(sinr_files)}.")

        rsrp, _, sinr = dl.load_preprocess_dataset(
            config,
            data_dir,
            rsrp_filename,
            sinr_filename,
            upsample_factor=1,  # Depending on the dataset
            transpose_result=True,
        )

        # Handover protocol
        env = HandoverEnv3GPP(config)

        # Simulate handover
        env.predict(rsrp, sinr)

        # Save statistics
        stats = env.get_statistics(sinr)
        for key, val in stats.items():
            aggregated_stats[key].append(val)

        # Save additional statistics
        result_container["sinr_connected"][speed].extend(env.sinr_timeline)
        result_container["sinr_max"][speed].extend(list(np.max(sinr, axis=0)))
        result_container["sinr_at_ho_exe_pcell"].extend(env.sinr_at_ho_exe_pcell)
        result_container["sinr_after_ho_exe_tcell"].extend(env.sinr_after_ho_exe_tcell)
        result_container["n_ho"][speed].append(stats["num_ho_exe_started"])
        result_container["n_pp"][speed].append(stats["num_pp"])
        result_container["n_rlf"][speed].append(stats["num_rlf"])

    # SINR (all speeds combined)
    sinr_at_ho_exe_pcell_db = np.array(result_container["sinr_at_ho_exe_pcell"])
    sinr_after_ho_exe_tcell_db = np.array(result_container["sinr_after_ho_exe_tcell"])
    sinr_at_ho_exe_pcell_db[np.isnan(sinr_at_ho_exe_pcell_db)] = -np.inf
    sinr_after_ho_exe_tcell_db[np.isnan(sinr_after_ho_exe_tcell_db)] = -np.inf

    # Results (all speeds individually)
    r_rel = []
    mean_pp_prob = []
    mean_rlf_prob = []
    for speed in np.unique(speeds):
        sinr_connected_db = np.array(result_container["sinr_connected"][speed])
        sinr_connected_lin = 10 ** (sinr_connected_db / 10)
        sinr_connected_lin[np.isnan(sinr_connected_lin)] = 0

        sinr_max_lin = 10 ** (np.array(result_container["sinr_max"][speed]) / 10)
        sinr_max_lin[np.isnan(sinr_max_lin)] = 0

        r_mean = np.mean(config.bw * np.log2(1 + sinr_connected_lin))
        r_max = np.mean(config.bw * np.log2(1 + sinr_max_lin))

        r_rel.append(r_mean / r_max)
        mean_pp_prob.append(
            np.mean(result_container["n_pp"][speed])
            / np.mean(result_container["n_ho"][speed])
        )
        mean_rlf_prob.append(
            np.mean(result_container["n_rlf"][speed])
            / np.mean(result_container["n_ho"][speed])
        )

    # Print aggregated statistics
    aggregated_stats["speeds"] = np.unique(speeds)
    aggregated_stats["r_rel"] = r_rel
    aggregated_stats["mean_pp_prob"] = mean_pp_prob
    aggregated_stats["mean_rlf_prob"] = mean_rlf_prob
    ut.print_aggregated_stats(aggregated_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
