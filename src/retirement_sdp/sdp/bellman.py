"""Backward induction (Bellman equation) — core SDP engine.

The value function V(W, t) satisfies:
    V(W, T)   = U(W)           terminal (bequest utility)
    V(0, t)   = -∞             absorbing ruin state
    V(W, t)   = max_{a ∈ A} { β · E[V(W', t+1)] }
where
    W' = (W + c_t) · R_p        (accumulation)
    W' = (W - d_t) · R_p        (drawdown, d_t ≥ 0)
    R_p ~ LogNormal(μ_p, σ_p²)  (portfolio gross return)

Implementation:
- Wealth grid: log-spaced, shape (n_W,)
- Action space: (n_A, n_assets) weight matrix
- Monte Carlo with S samples to estimate E[V(W', t+1)]
- Linear interpolation of V on the wealth grid
- Vectorised over wealth grid for speed
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.interpolate import interp1d

from .utility import crra_utility, _NEG_INF
from .action_space import build_efficient_frontier

logger = logging.getLogger(__name__)


def _interp_value(
    V_next: np.ndarray,
    grid: np.ndarray,
) -> interp1d:
    """Return a linear interpolant for V_next defined on *grid*.

    Outside the grid, clamp to the boundary values.
    """
    return interp1d(
        grid,
        V_next,
        kind="linear",
        bounds_error=False,
        fill_value=(V_next[0], V_next[-1]),
    )


def bellman_backward(
    grid: np.ndarray,          # (n_W,) wealth grid
    cash_flows: np.ndarray,    # (T,) per-period cash flows (+ contrib, - withdraw)
    return_model,              # LogNormalReturnModel
    weights: np.ndarray,       # (n_A, n_assets) efficient-frontier portfolios
    gamma: float,
    beta: float,
    mc_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run backward induction from T to 0.

    Returns
    -------
    V : (n_W, T+1) value function array
        V[:, t] is the value function at period t.
    policy : (n_W, T) optimal action index array
        policy[i, t] is the index into *weights* for wealth grid[i] at time t.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_W = len(grid)
    T = len(cash_flows)
    n_A = len(weights)

    V = np.full((n_W, T + 1), _NEG_INF)
    policy = np.zeros((n_W, T), dtype=np.int32)

    # Terminal: V(W, T) = U(W)
    V[:, T] = crra_utility(grid, gamma)

    t0 = time.time()
    for t in range(T - 1, -1, -1):
        cf = cash_flows[t]
        V_interp = _interp_value(V[:, t + 1], grid)

        # Adjusted wealth after cash flow (before investment return)
        # Shape: (n_W,)
        W_adj = grid + cf

        # Clamp: if W_adj ≤ 0, that node immediately ruins
        ruined = W_adj <= 0.0

        # For each action a, compute expected value via Monte Carlo
        # EV shape: (n_W, n_A)
        EV = np.full((n_W, n_A), _NEG_INF)

        for a_idx in range(n_A):
            w_a = weights[a_idx]
            # Draw portfolio gross returns
            R_samples = return_model.sample_returns(w_a, mc_samples, rng=rng)
            # W' = W_adj * R   shape: (n_W, S)
            W_next = np.outer(W_adj, R_samples)  # (n_W, S)
            W_next = np.clip(W_next, 0.0, None)  # floor at 0

            # Interpolate V(W', t+1) for each sample; mean over S
            # Process in slices to avoid huge memory: (n_W × S)
            V_next_vals = V_interp(W_next.ravel()).reshape(n_W, mc_samples)
            mean_V = V_next_vals.mean(axis=1)   # (n_W,)

            EV[:, a_idx] = mean_V

        # Ruin override: ruined nodes get -∞ regardless of action
        EV[ruined, :] = _NEG_INF

        # Optimal action
        best_a = np.argmax(EV, axis=1)           # (n_W,)
        best_ev = EV[np.arange(n_W), best_a]

        V[:, t] = beta * best_ev
        V[ruined, t] = _NEG_INF
        policy[:, t] = best_a

        if (T - t) % max(1, T // 10) == 0 or t == 0:
            elapsed = time.time() - t0
            logger.info(
                "Bellman t=%3d/%d | elapsed=%.1fs | "
                "V range=[%.2f, %.2f]",
                t,
                T,
                elapsed,
                np.percentile(V[V[:, t] > _NEG_INF / 2, t], 5) if np.any(V[:, t] > _NEG_INF / 2) else _NEG_INF,
                np.max(V[:, t]),
            )

    logger.info("Bellman backward complete in %.1fs.", time.time() - t0)
    return V, policy
