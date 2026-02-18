"""CRRA utility function.

U(W) = W^(1-γ) / (1-γ)   for γ ≠ 1
U(W) = ln(W)              for γ = 1

Properties:
- U is strictly concave (risk-averse)
- U(W) → -∞ as W → 0⁺   (ruin is infinitely bad)
- γ controls relative risk aversion (default 3.0)
"""

from __future__ import annotations

import numpy as np

_NEG_INF = -1e30   # Sentinel for ruin state (W ≤ 0)


def crra_utility(W: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Compute CRRA utility.

    Parameters
    ----------
    W : wealth level(s) — may be scalar or array
    gamma : risk-aversion coefficient (γ > 0, γ ≠ 1 for power form)

    Returns
    -------
    U(W) with -∞ sentinel for W ≤ 0
    """
    scalar = np.isscalar(W)
    W = np.atleast_1d(np.asarray(W, dtype=float))
    U = np.full_like(W, _NEG_INF)
    pos = W > 0
    if gamma == 1.0:
        U[pos] = np.log(W[pos])
    else:
        U[pos] = W[pos] ** (1.0 - gamma) / (1.0 - gamma)
    return float(U[0]) if scalar else U


def crra_marginal_utility(W: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """U'(W) = W^(-γ).  Returns 0 for W ≤ 0 (gradient not used there)."""
    scalar = np.isscalar(W)
    W = np.atleast_1d(np.asarray(W, dtype=float))
    dU = np.zeros_like(W)
    pos = W > 0
    dU[pos] = W[pos] ** (-gamma)
    return float(dU[0]) if scalar else dU
