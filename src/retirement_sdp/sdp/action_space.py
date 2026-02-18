"""Efficient-frontier portfolio discretisation via SLSQP.

Generates ~N_PORTFOLIOS mean-variance efficient portfolios by tracing the
frontier from minimum-variance to maximum-Sharpe to maximum-return.
Each portfolio is a weight vector w ∈ ℝ^N with w_i ≥ 0, Σw_i = 1.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    return float(w @ Sigma @ w)


def _portfolio_variance_grad(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    return 2.0 * Sigma @ w


def _min_variance_portfolio(
    Sigma: np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Find the global minimum-variance portfolio."""
    w0 = np.ones(n_assets) / n_assets
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_assets
    result = minimize(
        _portfolio_variance,
        w0,
        args=(Sigma,),
        jac=_portfolio_variance_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        logger.warning("Min-variance optimisation did not converge: %s", result.message)
    return result.x


def _portfolio_for_target_return(
    target_mu: float,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> np.ndarray:
    """Minimum-variance portfolio for a given target return (long-only)."""
    n = len(mu)
    w0 = np.ones(n) / n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_mu},
    ]
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        _portfolio_variance,
        w0,
        args=(Sigma,),
        jac=_portfolio_variance_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        logger.debug(
            "SLSQP did not converge for target_mu=%.4f: %s",
            target_mu,
            result.message,
        )
    return np.clip(result.x, 0.0, 1.0)


def build_efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_portfolios: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute efficient frontier portfolios.

    Parameters
    ----------
    mu : (N,) annualised mean log-return vector
    Sigma : (N, N) annualised covariance matrix
    n_portfolios : number of frontier portfolios to generate

    Returns
    -------
    weights : (n_portfolios, N) weight matrix — each row is a portfolio
    frontier_mu : (n_portfolios,) expected return for each portfolio
    frontier_sigma : (n_portfolios,) volatility for each portfolio
    """
    n = len(mu)
    mu_min_var = float(_min_variance_portfolio(Sigma, n) @ mu)
    mu_max = float(np.max(mu))

    if mu_max <= mu_min_var:
        target_returns = np.linspace(mu_min_var, mu_min_var + 1e-6, n_portfolios)
    else:
        target_returns = np.linspace(mu_min_var, mu_max, n_portfolios)

    weights = []
    for target in target_returns:
        w = _portfolio_for_target_return(target, mu, Sigma)
        # normalise to handle numerical drift
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s > 0:
            w /= s
        weights.append(w)

    weights = np.array(weights)
    frontier_mu = weights @ mu
    frontier_sigma = np.sqrt(
        np.array([w @ Sigma @ w for w in weights])
    )

    # Remove duplicate portfolios (numerical noise)
    _, unique_idx = np.unique(np.round(frontier_mu, 6), return_index=True)
    weights = weights[unique_idx]
    frontier_mu = frontier_mu[unique_idx]
    frontier_sigma = frontier_sigma[unique_idx]

    logger.info(
        "Efficient frontier: %d unique portfolios, mu=[%.4f, %.4f], sigma=[%.4f, %.4f]",
        len(weights),
        frontier_mu.min(),
        frontier_mu.max(),
        frontier_sigma.min(),
        frontier_sigma.max(),
    )
    return weights, frontier_mu, frontier_sigma
