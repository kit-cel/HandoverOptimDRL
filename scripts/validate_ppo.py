"""Validate PPO protocol on the handover environment."""

import os

import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
from ho_optim_drl.gym_env.ho_env_ppo import test_ppo_model
import ho_optim_drl.utils as ut


def main(root_path: str):
    """Validate PPO on the handover environment."""
    # Load configuration
    config = Config()

    # Load MATLAB files
    data_dir = os.path.join(root_path, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")

    # Speed filter
    use_speed_list = [30]
    rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(
        rsrp_files, sinr_files, use_speed_list
    )

    # Load all datasets
    rsrp_list = []
    rsrq_list = []
    sinr_list = []
    for rsrp_fname_i, sinr_fname_i in zip(rsrp_files, sinr_files):
        # Load dataset
        rsrp, rsrq, sinr = dl.load_preprocess_dataset(
            config, data_dir, rsrp_fname_i, sinr_fname_i
        )

        # Clip and normalize
        rsrq = ut.norm(rsrq, config.clip_l, config.clip_h)

        sinr_list.append(sinr)
        rsrp_list.append(rsrp)
        rsrq_list.append(rsrq)

    # Generate environment
    env = HandoverEnvPPO(config, rsrp_list, rsrq_list, sinr_list)
    check_env(env, warn=True)

    # Load PPO model
    model_dir = os.path.join(root_path, "results", "models", "ppo_model", "model")
    model = PPO.load(
        model_dir,
        env=env,
        tensorboard_log=None,
    )

    # Result containers
    result_container = ut.get_result_container(speeds)
    aggregated_stats = {key: [] for key in env.ho_procedure.get_stats_dict()}

    # Test PPO model on environment
    if config.test_deterministic_actions:
        print("[PPO] Test with deterministic actions.")
    else:
        print("[PPO] Test with actions sampled from the policy distribution.")
    print(f"[Env] HO preparation abort permitted: {config.permit_ho_prep_abort}")
    for i in range(env.n_datasets):
        # Test the model on the environment
        test_ppo_model(env, model, i)
        print(f"Testing PPO HO protocol with dataset {i + 1:3d}/{env.n_datasets}.")

        # Save statistics
        stats = env.get_statistics()
        for key, val in stats.items():
            aggregated_stats[key].append(val)

        # Save additional statistics
        result_container["sinr_connected"][speeds[i]].extend(
            env.ho_procedure.sinr_timeline
        )
        result_container["sinr_max"][speeds[i]].extend(
            list(np.max(env.sinr_list[env.dataset_idx], axis=1))
        )
        result_container["sinr_at_ho_exe_pcell"].extend(
            env.ho_procedure.sinr_at_ho_exe_pcell
        )
        result_container["sinr_after_ho_exe_tcell"].extend(
            env.ho_procedure.sinr_after_ho_exe_tcell
        )
        result_container["n_ho"][speeds[i]].append(stats["num_ho_exe_started"])
        result_container["n_pp"][speeds[i]].append(stats["num_pp"])
        result_container["n_rlf"][speeds[i]].append(stats["num_rlf"])

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
    aggregated_stats["speeds"] = np.unique(speeds).tolist()
    aggregated_stats["r_rel"] = r_rel
    aggregated_stats["mean_pp_prob"] = mean_pp_prob
    aggregated_stats["mean_rlf_prob"] = mean_rlf_prob
    ut.print_aggregated_stats(aggregated_stats)
