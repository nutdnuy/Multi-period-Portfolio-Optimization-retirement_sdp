"""Multi-panel wealth PDF plot over time."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.pdf_propagation import PeriodPDF


def plot_wealth_pdfs(
    period_pdfs: Sequence[PeriodPDF],
    w_grid: np.ndarray,
    output_path: str | Path,
    dpi: int = 150,
    title: str = "Wealth Distribution Over Time",
) -> None:
    """Plot per-period wealth PDFs in a multi-panel figure.

    Parameters
    ----------
    period_pdfs : sequence of PeriodPDF objects (one per panel)
    w_grid : wealth values at which to evaluate PDFs
    output_path : save destination
    dpi : figure DPI
    title : overall figure title
    """
    n = len(period_pdfs)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=False
    )
    axes = np.array(axes).ravel()

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.15, 0.85, n))

    for i, (pdf_obj, ax) in enumerate(zip(period_pdfs, axes)):
        density = pdf_obj.evaluate_on_grid(w_grid)
        ax.fill_between(w_grid / 1e6, density * 1e6, alpha=0.4, color=colors[i])
        ax.plot(w_grid / 1e6, density * 1e6, color=colors[i], linewidth=1.5)

        # Ruin annotation
        ruin_pct = pdf_obj.ruin_fraction * 100
        ax.set_title(
            f"t={pdf_obj.t}  {pdf_obj.phase_label}\nRuin: {ruin_pct:.1f}%",
            fontsize=9,
        )
        ax.set_xlabel("Wealth ($M)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # Add text showing ruin fraction as shaded area on the left
        if pdf_obj.ruin_fraction > 0:
            ax.text(
                0.02, 0.97,
                f"P(ruin)={ruin_pct:.1f}%",
                transform=ax.transAxes,
                fontsize=7, va="top", color="red",
            )

    # Hide unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
