"""Ruin probability vs withdrawal rate curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_ruin_curve(
    withdrawal_rates: np.ndarray,
    ruin_probabilities: np.ndarray,
    swr: float,
    swr_ruin: float,
    tolerance: float,
    output_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot ruin probability as a function of withdrawal rate.

    Parameters
    ----------
    withdrawal_rates : (K,) withdrawal rates evaluated
    ruin_probabilities : (K,) corresponding ruin probabilities
    swr : computed safe withdrawal rate
    swr_ruin : ruin probability at SWR
    tolerance : the tolerance threshold
    output_path : save path
    dpi : output DPI
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Main curve
    ax.plot(
        withdrawal_rates * 100,
        ruin_probabilities * 100,
        "b-o",
        linewidth=2,
        markersize=5,
        label="P(ruin) vs withdrawal rate",
    )

    # Tolerance line
    ax.axhline(
        y=tolerance * 100,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Tolerance = {tolerance*100:.0f}%",
    )

    # SWR vertical line
    ax.axvline(
        x=swr * 100,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"SWR = {swr*100:.2f}%  (P_ruin={swr_ruin*100:.1f}%)",
    )

    # Shaded safe zone
    ax.axvspan(0, swr * 100, alpha=0.08, color="green", label="Safe zone")

    # Annotate SWR
    ax.annotate(
        f"SWR = {swr*100:.2f}%",
        xy=(swr * 100, swr_ruin * 100),
        xytext=(swr * 100 + 0.3, swr_ruin * 100 + 5),
        fontsize=10,
        color="green",
        arrowprops={"arrowstyle": "->", "color": "green"},
    )

    ax.set_xlabel("Annual Withdrawal Rate (% of initial wealth)", fontsize=12)
    ax.set_ylabel("Lifetime Ruin Probability (%)", fontsize=12)
    ax.set_title("Ruin Probability vs. Withdrawal Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, min(100, ruin_probabilities.max() * 100 * 1.15 + 5))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
