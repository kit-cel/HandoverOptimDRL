"""Validate 3GPP protocol on the handover environment."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import colors


def load_sinr_column(csv_path: str, col: str) -> np.ndarray:
    """Load one SINR column from a semicolon-separated CSV."""
    df = pd.read_csv(csv_path, sep=";")
    if col not in df.columns:
        raise ValueError(f'Missing column "{col}" in {csv_path}')
    values = df[col].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f'Column "{col}" in {csv_path} is empty.')
    return values


def plot_ho_sinr_ecdf(root_path: str, q_out: float = -8.0, q_in: float = -6.0) -> None:
    """
    Plot the empirical CDF of the SINR before the handover (HO) execution in the
    serving cell and the SINR of the traget cell (new serving cell) after the HO.
    """
    ref_pre_path = os.path.join(root_path, "results", "ho_sinr", "ref_pre_ho_sinr.csv")
    ref_post_path = os.path.join(
        root_path, "results", "ho_sinr", "ref_post_ho_sinr.csv"
    )
    ppo_pre_path = os.path.join(root_path, "results", "ho_sinr", "ppo_pre_ho_sinr.csv")
    ppo_post_path = os.path.join(
        root_path, "results", "ho_sinr", "ppo_post_ho_sinr.csv"
    )

    ref_pre = load_sinr_column(ref_pre_path, col="pre_ho_sinr_db")
    ref_post = load_sinr_column(ref_post_path, col="post_ho_sinr_db")
    ppo_pre = load_sinr_column(ppo_pre_path, col="pre_ho_sinr_db")
    ppo_post = load_sinr_column(ppo_post_path, col="post_ho_sinr_db")

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=150)

    # Curves
    ax.ecdf(
        ref_pre,
        linewidth=1.4,
        linestyle="solid",
        color=colors.KIT_BLUE,
        label="3GPP before HO",
    )
    ax.ecdf(
        ref_post,
        linewidth=1.4,
        linestyle="dashed",
        color=colors.KIT_BLUE,
        label="3GPP after HO",
    )
    ax.ecdf(
        ppo_pre,
        linewidth=1.4,
        linestyle="solid",
        color=colors.KIT_ORANGE,
        label="PPO before HO",
    )
    ax.ecdf(
        ppo_post,
        linewidth=1.4,
        linestyle="dashed",
        color=colors.KIT_ORANGE,
        label="PPO after HO",
    )

    ax.axvline(
        q_out,
        linewidth=1.2,
        linestyle="solid",
        color=colors.KIT_RED,
        label=r"$Q_{\mathrm{out}}$",
    )
    ax.axvline(
        q_in,
        linewidth=1.2,
        linestyle="dashed",
        color=colors.KIT_RED,
        label=r"$Q_{\mathrm{in}}$",
    )

    ax.set_xlabel(r"SINR (dB)")
    ax.set_ylabel(r"ECDF")

    ax.set_xlim(-15.0, 10.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([-15, -10, -5, 0, 5, 10])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=2.2,
        handletextpad=0.5,
        borderpad=0.5,
    )

    fig.tight_layout()

    out_file = "ho_sinr_ecdf.png"
    out_dir = os.path.join(root_path, "results", "plots")
    out_path = os.path.join(out_dir, out_file)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_path}")
