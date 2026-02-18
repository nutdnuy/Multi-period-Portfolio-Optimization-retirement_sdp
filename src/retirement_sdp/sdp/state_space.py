"""Log-spaced wealth grid for the SDP state space."""

from __future__ import annotations

import numpy as np


def build_wealth_grid(
    w_min: float,
    w_max: float,
    n_points: int,
) -> np.ndarray:
    """Build a log-spaced wealth grid with *n_points* nodes in [w_min, w_max].

    Log-spacing puts more resolution near zero (where utility curvature is
    greatest under CRRA) and less near the upper end.

    Parameters
    ----------
    w_min : minimum wealth node (must be > 0)
    w_max : maximum wealth node
    n_points : number of grid nodes

    Returns
    -------
    (n_points,) sorted array of wealth values
    """
    if w_min <= 0:
        raise ValueError(f"w_min must be > 0, got {w_min}")
    return np.logspace(np.log10(w_min), np.log10(w_max), n_points)


def nearest_grid_index(grid: np.ndarray, value: float) -> int:
    """Return the index of the grid node nearest to *value*."""
    return int(np.argmin(np.abs(grid - value)))
