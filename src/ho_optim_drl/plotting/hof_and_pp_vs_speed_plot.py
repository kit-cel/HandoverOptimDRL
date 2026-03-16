"""Plot the ping-pong (PP)-rate and the handover failure (HOF)-rate vs. the UE speeds."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import colors


def plot_hof_and_pp_vs_speed(
    ref_df: pd.DataFrame,
    ppo_df: pd.DataFrame,
    root_path: str,
) -> None:
    """Plot the ping-pong (PP)-rate and the handover failure (HOF)-rate vs. the UE speeds."""

    speeds = sorted(set(ref_df["speed"]).intersection(ppo_df["speed"]))
    if not speeds:
        raise ValueError("No common speed values found in both CSV files.")

    ref_df = ref_df.set_index("speed").loc[speeds].reset_index()
    ppo_df = ppo_df.set_index("speed").loc[speeds].reset_index()

    x = np.arange(len(speeds), dtype=float)
    width = 0.12

    fig, ax_hof = plt.subplots(figsize=(5.0, 4.2), dpi=150)
    ax_pp = ax_hof.twinx()

    # PP bars (right axis)
    pp_ref = ax_pp.bar(
        x - 0.5 * width,
        ref_df["mean_pp_rate"],
        width=width,
        label=r"PP$_{\mathrm{3GPP}}$",
        edgecolor=colors.KIT_BLUE,
        linewidth=0.6,
        hatch="///",
        fill=False,
    )
    pp_ppo = ax_pp.bar(
        x + 0.5 * width,
        ppo_df["mean_pp_rate"],
        width=width,
        label=r"PP$_{\mathrm{PPO}}$",
        edgecolor=colors.KIT_ORANGE,
        linewidth=0.6,
        hatch="///",
        fill=False,
    )
    pp_ref = ax_pp.bar(
        x - 0.5 * width,
        ref_df["mean_pp_rate"],
        width=width,
        edgecolor="black",
        linewidth=0.6,
        fill=False,
    )
    pp_ppo = ax_pp.bar(
        x + 0.5 * width,
        ppo_df["mean_pp_rate"],
        width=width,
        edgecolor="black",
        linewidth=0.6,
        fill=False,
    )

    # HOF bars (left axis)
    hof_ref = ax_hof.bar(
        x - 0.5 * width,
        ref_df["mean_rlf_rate"],
        width=width,
        label=r"HOF$_{\mathrm{3GPP}}$",
        facecolor=colors.KIT_BLUE,
        edgecolor="black",
        linewidth=0.6,
    )
    hof_ppo = ax_hof.bar(
        x + 0.5 * width,
        ppo_df["mean_rlf_rate"],
        width=width,
        label=r"HOF$_{\mathrm{PPO}}$",
        facecolor=colors.KIT_ORANGE,
        edgecolor="black",
        linewidth=0.6,
    )

    ax_hof.set_xlabel("UE velocity (km/h)")
    ax_hof.set_ylabel("HOF probability")
    ax_pp.set_ylabel("PP probability")

    ax_hof.set_xticks(x)
    ax_hof.set_xticklabels([str(int(v)) for v in speeds])

    ax_hof.set_ylim(0.0, 0.15)
    ax_pp.set_ylim(0.0, 0.60)

    ax_hof.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax_pp.set_yticks([0.0, 0.2, 0.4, 0.6])

    ax_hof.grid(axis="y", alpha=0.3)
    ax_hof.set_axisbelow(True)

    handles = [hof_ref, hof_ppo, pp_ref, pp_ppo]
    labels = [
        r"HOF$_{\mathrm{3GPP}}$",
        r"HOF$_{\mathrm{PPO}}$",
        r"PP$_{\mathrm{3GPP}}$",
        r"PP$_{\mathrm{PPO}}$",
    ]
    ax_hof.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=0.8,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    fig.tight_layout()

    out_file = "hof_pp_plot.png"
    out_dir = os.path.join(root_path, "results", "plots")
    out_path = os.path.join(out_dir, out_file)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_path}")
