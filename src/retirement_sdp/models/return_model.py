"""Log-normal return model: fit (mu, Sigma) from historical data.

For a portfolio with weight vector w, the one-period log-return is:
    r_p ~ Normal(mu_p, sigma_p^2)
where
    mu_p    = w^T mu  - 0.5 * w^T Sigma w      (Ito correction for log-normal)
    sigma_p = sqrt(w^T Sigma w)

Portfolio gross return: R_p = exp(r_p) ~ LogNormal(mu_p, sigma_p^2)
"""

from __future__ import annotations

import numpy as np


class LogNormalReturnModel:
    """Fitted log-normal return model for a multi-asset portfolio.

    Parameters
    ----------
    mu : (N,) annualised mean log-returns (from historical data)
    Sigma : (N, N) annualised covariance matrix
    tickers : list of asset names (for display only)
    """

    def __init__(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        tickers: list[str] | None = None,
    ) -> None:
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.tickers = tickers or [f"asset_{i}" for i in range(len(mu))]
        self._n = len(mu)

    # ------------------------------------------------------------------
    # Portfolio moments
    # ------------------------------------------------------------------

    def portfolio_log_mean(self, w: np.ndarray) -> float:
        """mu_p = w^T mu  - 0.5 * w^T Sigma w  (Ito-corrected log-mean)."""
        w = np.asarray(w, dtype=float)
        return float(w @ self.mu - 0.5 * w @ self.Sigma @ w)

    def portfolio_log_variance(self, w: np.ndarray) -> float:
        """sigma_p^2 = w^T Sigma w."""
        w = np.asarray(w, dtype=float)
        return float(w @ self.Sigma @ w)

    def portfolio_log_std(self, w: np.ndarray) -> float:
        return float(np.sqrt(self.portfolio_log_variance(w)))

    def portfolio_arithmetic_mean(self, w: np.ndarray) -> float:
        """E[R_p] = exp(mu_p + 0.5 * sigma_p^2)."""
        mu_p = self.portfolio_log_mean(w)
        sigma2_p = self.portfolio_log_variance(w)
        return float(np.exp(mu_p + 0.5 * sigma2_p))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_returns(
        self,
        w: np.ndarray,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw *n_samples* gross returns R_p = exp(r_p) for weight vector *w*.

        Parameters
        ----------
        w : (N,) portfolio weight vector (must sum to 1)
        n_samples : number of i.i.d. samples
        rng : numpy random Generator (for reproducibility)

        Returns
        -------
        (n_samples,) array of gross returns
        """
        if rng is None:
            rng = np.random.default_rng()
        mu_p = self.portfolio_log_mean(w)
        sigma_p = self.portfolio_log_std(w)
        log_returns = rng.normal(mu_p, sigma_p, size=n_samples)
        return np.exp(log_returns)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["LogNormalReturnModel"]
        for i, t in enumerate(self.tickers):
            lines.append(
                f"  {t}: mu={self.mu[i]:.4f}  sigma={np.sqrt(self.Sigma[i,i]):.4f}"
            )
        return "\n".join(lines)
