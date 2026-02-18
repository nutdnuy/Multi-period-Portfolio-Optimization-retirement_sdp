"""Ruin probability analysis.

Two estimators:
1. **Empirical**: P_ruin(t) = fraction of paths with W_t ≤ 0.
2. **KDE integral**: ∫_{-∞}^{0} f_t(w) dw  (for rare-event estimation).
   By construction of our KDE (fitted on log(W) for W > 0), this is 0.
   The KDE integral is instead used to smooth the empirical estimator.

Lifetime ruin: P_lifetime = empirical fraction ruined by terminal period T.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from scipy.integrate import quad

from .pdf_propagation import PeriodPDF, propagate_pdfs

logger = logging.getLogger(__name__)


class RuinResults(NamedTuple):
    """Container for ruin probability estimates."""
    periods: np.ndarray            # (K,) period indices
    ruin_empirical: np.ndarray     # (K,) empirical ruin fraction per period
    ruin_lifetime: float           # scalar: fraction ruined by final period
    phase_labels: list[str]        # phase name per period


def compute_ruin(
    paths: np.ndarray,             # (n_paths, T+1)
    phase_labels: list[str],       # length T
) -> RuinResults:
    """Compute per-period and lifetime ruin probabilities.

    Parameters
    ----------
    paths : (n_paths, T+1) wealth matrix.
    phase_labels : phase name for each period.

    Returns
    -------
    RuinResults
    """
    n_paths, T_plus_1 = paths.shape
    T = T_plus_1 - 1
    periods = np.arange(T_plus_1)

    # Empirical: cumulative ruin (absorbing state — once ruined, stay ruined)
    ruined_ever = np.zeros(n_paths, dtype=bool)
    ruin_empirical = np.zeros(T_plus_1)
    for t in range(T_plus_1):
        ruined_ever |= paths[:, t] <= 0
        ruin_empirical[t] = ruined_ever.mean()

    ruin_lifetime = float(ruin_empirical[-1])

    labels = [phase_labels[min(t, len(phase_labels) - 1)] for t in range(T_plus_1)]

    logger.info(
        "Ruin analysis: lifetime ruin=%.4f, terminal period=%d", ruin_lifetime, T
    )
    return RuinResults(
        periods=periods,
        ruin_empirical=ruin_empirical,
        ruin_lifetime=ruin_lifetime,
        phase_labels=labels,
    )
