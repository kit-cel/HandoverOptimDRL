"""Validate 3GPP protocol on the handover environment."""

import os


import ho_optim_drl.utils as ut
import ho_optim_drl.plotting as ho_plt


def main(
    root_path: str, ref_file: str | None = None, ppo_file: str | None = None
) -> int:
    """Main script to reproduce all figures."""

    ref_file = "ref_metrics.csv" if ref_file is None else ref_file
    ppo_file = "ppo_metrics.csv" if ppo_file is None else ppo_file

    ref_path = os.path.join(root_path, "results", "metrics", ref_file)
    ppo_path = os.path.join(root_path, "results", "metrics", ppo_file)

    ref_df = ut.load_metrics(ref_path)
    ppo_df = ut.load_metrics(ppo_path)

    ho_plt.plot_hof_and_pp_vs_speed(ref_df, ppo_df, root_path)
    ho_plt.plot_relative_achieved_rate_vs_speed(ref_df, ppo_df, root_path)
    ho_plt.plot_ho_sinr_ecdf(root_path)

    return 0
