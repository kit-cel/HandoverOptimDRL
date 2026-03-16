"""Plot the relative achieved rate vs. the UE speeds."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import colors


def plot_relative_achieved_rate_vs_speed(
    ref_df: pd.DataFrame,
    ppo_df: pd.DataFrame,
    root_path: str,
) -> None:
    """Plot the relative achieved rate vs. the UE speeds."""
    speeds = sorted(set(ref_df["speed"]).intersection(ppo_df["speed"]))
    if not speeds:
        raise ValueError("No common speed values found in both CSV files.")

    ref_df = ref_df.set_index("speed").loc[speeds].reset_index()
    ppo_df = ppo_df.set_index("speed").loc[speeds].reset_index()

    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=150)

    ax.plot(
        speeds,
        100 * ppo_df["r_rel"],
        marker="o",
        markersize=6,
        linewidth=1.2,
        color=colors.KIT_ORANGE,
        markerfacecolor="none",
        label=r"$T_R$ PPO",
    )

    ax.plot(
        speeds,
        100 * ref_df["r_rel"],
        marker="^",
        markersize=6,
        linewidth=1.2,
        color=colors.KIT_BLUE,
        markerfacecolor="none",
        label=r"$T_R$ 3GPP",
    )

    ax.set_xlabel(r"UE velocity (km/h)")
    ax.set_ylabel(r"$\Gamma_\text{R}$ (%)")

    ax.set_xlim(min(speeds) - 5, max(speeds) + 5)
    ax.set_ylim(99.0, 100.0)

    ax.set_xticks(speeds)
    ax.set_yticks(np.linspace(99.0, 100.0, 6))

    ax.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    fig.tight_layout()

    out_file = "rel_achieved_rate_plot.png"
    out_dir = os.path.join(root_path, "results", "plots")
    out_path = os.path.join(out_dir, out_file)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_path}")
