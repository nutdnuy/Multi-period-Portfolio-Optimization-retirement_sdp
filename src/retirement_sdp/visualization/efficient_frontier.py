"""Efficient frontier plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_efficient_frontier(
    frontier_mu: np.ndarray,
    frontier_sigma: np.ndarray,
    weights: np.ndarray,
    tickers: list[str],
    individual_mu: np.ndarray,
    individual_sigma: np.ndarray,
    output_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot and save the mean-variance efficient frontier.

    Parameters
    ----------
    frontier_mu : (n_portfolios,) expected return for each frontier portfolio
    frontier_sigma : (n_portfolios,) volatility for each frontier portfolio
    weights : (n_portfolios, n_assets) weight matrix
    tickers : asset names
    individual_mu : (n_assets,) individual asset expected returns
    individual_sigma : (n_assets,) individual asset volatilities
    output_path : save destination
    dpi : output DPI
    """
    fig, (ax_main, ax_weights) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # ── Frontier curve ────────────────────────────────────────────────
    sc = ax_main.scatter(
        frontier_sigma * 100,
        frontier_mu * 100,
        c=frontier_mu / np.maximum(frontier_sigma, 1e-9),   # Sharpe ratio color
        cmap="viridis",
        s=20,
        zorder=3,
        label="Efficient frontier",
    )
    plt.colorbar(sc, ax=ax_main, label="Sharpe ratio (approx.)")

    # Individual assets
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    for i, (t, mu_i, sig_i) in enumerate(zip(tickers, individual_mu, individual_sigma)):
        ax_main.scatter(
            sig_i * 100,
            mu_i * 100,
            marker="*",
            s=180,
            zorder=5,
            color=colors[i],
            label=t,
        )

    ax_main.set_xlabel("Annualised Volatility (%)")
    ax_main.set_ylabel("Annualised Log-Return (%)")
    ax_main.set_title("Mean-Variance Efficient Frontier")
    ax_main.legend(loc="lower right", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # ── Weight stacked bar (sample 10 portfolios evenly) ─────────────
    idx = np.linspace(0, len(weights) - 1, min(10, len(weights)), dtype=int)
    w_sample = weights[idx]
    mu_sample = frontier_mu[idx] * 100

    bottom = np.zeros(len(idx))
    bar_colors = plt.cm.Set2(np.linspace(0, 1, len(tickers)))
    for a_i, ticker in enumerate(tickers):
        ax_weights.bar(
            range(len(idx)),
            w_sample[:, a_i],
            bottom=bottom,
            label=ticker,
            color=bar_colors[a_i],
        )
        bottom += w_sample[:, a_i]

    ax_weights.set_xticks(range(len(idx)))
    ax_weights.set_xticklabels([f"{m:.1f}%" for m in mu_sample], rotation=45, fontsize=8)
    ax_weights.set_xlabel("Expected Return (portfolio sample)")
    ax_weights.set_ylabel("Portfolio Weight")
    ax_weights.set_title("Frontier Portfolio Compositions")
    ax_weights.legend(loc="upper left", fontsize=8)
    ax_weights.set_ylim(0, 1)
    ax_weights.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
