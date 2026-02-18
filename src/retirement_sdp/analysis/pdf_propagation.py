"""KDE wealth PDF at each period from Monte Carlo paths.

For each time period t:
1. Separate ruined paths (W = 0) from surviving paths (W > 0).
2. Fit a Gaussian KDE (Scott bandwidth) on log(W) for surviving paths.
3. Return ruin fractions and KDE objects for downstream analysis.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


class PeriodPDF:
    """Wrapper holding KDE and metadata for one time period."""

    def __init__(
        self,
        t: int,
        surviving_wealth: np.ndarray,
        ruin_fraction: float,
        phase_label: str = "",
    ) -> None:
        self.t = t
        self.phase_label = phase_label
        self.ruin_fraction = ruin_fraction
        self.n_surviving = len(surviving_wealth)

        if self.n_surviving >= 2:
            log_w = np.log(surviving_wealth)
            # KDE requires non-zero variance; skip if all values are identical
            if log_w.std() < 1e-12:
                self._kde = None
            else:
                self._kde = gaussian_kde(log_w, bw_method="scott")
            self._log_w_min = log_w.min()
            self._log_w_max = log_w.max()
        else:
            self._kde = None
            self._log_w_min = 0.0
            self._log_w_max = 1.0

    def pdf(self, W: np.ndarray) -> np.ndarray:
        """Evaluate the KDE-based PDF at wealth values *W*.

        Returns zeros for W ≤ 0 or when KDE is unavailable.
        """
        W = np.asarray(W, dtype=float)
        result = np.zeros_like(W)
        if self._kde is None:
            return result
        pos = W > 0
        if np.any(pos):
            log_w = np.log(W[pos])
            # KDE is on log-scale; transform to wealth-scale density
            result[pos] = self._kde(log_w) / W[pos]
        return result

    def evaluate_on_grid(self, w_grid: np.ndarray) -> np.ndarray:
        """Convenience wrapper for pdf evaluated on *w_grid*."""
        return self.pdf(w_grid)

    def ruin_probability_kde(self) -> float:
        """Integral of KDE below 0 in log-space (always 0 by construction)."""
        # KDE is fitted on log(W) for W > 0 → no mass below 0 in wealth space
        return 0.0


def propagate_pdfs(
    paths: np.ndarray,          # (n_paths, T+1)
    phase_labels: list[str],    # length T
    periods_to_sample: list[int] | None = None,
) -> list[PeriodPDF]:
    """Build PeriodPDF objects for each period (or a subset).

    Parameters
    ----------
    paths : (n_paths, T+1) wealth matrix from forward simulation.
    phase_labels : phase name for each period 0 … T-1.
    periods_to_sample : periods to analyse; defaults to all T+1 periods.

    Returns
    -------
    List of PeriodPDF objects, one per period.
    """
    n_paths, T_plus_1 = paths.shape
    T = T_plus_1 - 1

    if periods_to_sample is None:
        periods_to_sample = list(range(T_plus_1))

    results: list[PeriodPDF] = []
    for t in periods_to_sample:
        W_t = paths[:, t]
        alive_mask = W_t > 0
        ruin_frac = float((~alive_mask).mean())
        surviving = W_t[alive_mask]
        label = phase_labels[min(t, len(phase_labels) - 1)] if phase_labels else ""
        results.append(PeriodPDF(t, surviving, ruin_frac, label))
        logger.debug(
            "Period %d: ruin=%.4f, n_surviving=%d", t, ruin_frac, len(surviving)
        )

    return results
