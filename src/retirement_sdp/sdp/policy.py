"""Forward simulation under the optimal SDP policy.

Simulates *n_paths* wealth trajectories from t=0 to T, following the
optimal action (portfolio weights) selected by the pre-computed policy table.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def _get_action_index(
    policy: np.ndarray,   # (n_W, T)
    grid: np.ndarray,     # (n_W,)
    W: np.ndarray,        # (n_paths,) current wealth
    t: int,
) -> np.ndarray:
    """Look up optimal action index for each path at time t.

    Uses nearest-grid-point lookup (fast and sufficient given 500-node grid).
    """
    # Clip wealth to grid bounds before lookup
    W_clipped = np.clip(W, grid[0], grid[-1])
    # Nearest grid point via searchsorted
    idx = np.searchsorted(grid, W_clipped)
    idx = np.clip(idx, 0, len(grid) - 1)
    return policy[idx, t]


def forward_simulate(
    initial_wealth: float,
    cash_flows: np.ndarray,     # (T,)
    policy: np.ndarray,         # (n_W, T)
    grid: np.ndarray,           # (n_W,) wealth grid
    weights: np.ndarray,        # (n_A, n_assets)
    return_model,               # LogNormalReturnModel
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Forward-simulate wealth paths under the optimal policy.

    Parameters
    ----------
    initial_wealth : W_0
    cash_flows : (T,) array — positive = contribution, negative = withdrawal
    policy : (n_W, T) optimal action index from backward induction
    grid : (n_W,) wealth grid
    weights : (n_A, n_assets) efficient-frontier portfolio weights
    return_model : LogNormalReturnModel
    n_paths : number of Monte Carlo paths
    rng : numpy Generator

    Returns
    -------
    paths : (n_paths, T+1) wealth array
        paths[:, 0] = initial_wealth
        paths[:, t] = wealth at start of period t
        Ruined paths have W = 0 and stay at 0 (absorbing state).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = len(cash_flows)
    paths = np.zeros((n_paths, T + 1))
    paths[:, 0] = initial_wealth

    # Track which paths are ruined
    alive = np.ones(n_paths, dtype=bool)

    for t in range(T):
        W = paths[:, t].copy()
        cf = cash_flows[t]

        # Adjust for cash flow
        W_adj = W + cf
        ruined_now = W_adj <= 0.0
        alive &= ~ruined_now
        W_adj = np.maximum(W_adj, 0.0)

        # Determine optimal portfolio for each alive path
        a_idx = _get_action_index(policy, grid, W_adj, t)   # (n_paths,)

        # Draw portfolio returns — one per path
        W_next = np.zeros(n_paths)
        if np.any(alive):
            for a in np.unique(a_idx[alive]):
                mask = alive & (a_idx == a)
                if not np.any(mask):
                    continue
                n_mask = int(mask.sum())
                R = return_model.sample_returns(weights[a], n_mask, rng=rng)
                W_next[mask] = W_adj[mask] * R

        # Zero out ruined paths
        W_next[~alive] = 0.0
        paths[:, t + 1] = W_next

    logger.info(
        "Forward sim: %d paths, T=%d, final ruin fraction=%.4f",
        n_paths,
        T,
        1.0 - alive.mean(),
    )
    return paths
