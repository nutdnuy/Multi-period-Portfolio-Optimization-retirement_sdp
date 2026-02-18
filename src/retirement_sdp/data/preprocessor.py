"""Data preprocessing: log-returns, alignment, annualisation.

Takes raw monthly-adjusted price dicts (from AlphaVantageClient) and
returns annualised log-return DataFrames aligned across all tickers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def monthly_prices_to_series(raw: dict[str, dict]) -> pd.Series:
    """Convert Alpha Vantage monthly-adjusted dict to a price Series.

    Parameters
    ----------
    raw:
        Mapping ``"YYYY-MM-DD"`` → dict from TIME_SERIES_MONTHLY_ADJUSTED.

    Returns
    -------
    pd.Series with DatetimeIndex sorted ascending, values = adjusted close.
    """
    records = {
        pd.Timestamp(date): float(vals["5. adjusted close"])
        for date, vals in raw.items()
    }
    s = pd.Series(records, name="adj_close").sort_index()
    return s


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute monthly log-returns from a price series."""
    lr = np.log(prices / prices.shift(1)).dropna()
    lr.name = prices.name
    return lr


def align_returns(*series: pd.Series) -> pd.DataFrame:
    """Inner-join multiple return series on their DatetimeIndex."""
    df = pd.concat(series, axis=1, join="inner")
    df.index.name = "date"
    return df.dropna()


def annualise_returns(monthly_log_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert monthly log-return DataFrame to annualised (mu, Sigma).

    Uses standard annualisation:
    - mu_ann  = 12 × mean(monthly log-return)
    - Sigma_ann = 12 × cov(monthly log-returns)

    Returns
    -------
    mu : (N,) array of annualised mean log-returns
    Sigma : (N, N) annualised covariance matrix
    tickers : list[str]
    """
    mu = monthly_log_returns.mean().values * 12
    Sigma = monthly_log_returns.cov().values * 12
    return mu, Sigma


def build_price_dataframe(
    raw_prices: dict[str, dict],
    tickers: list[str],
) -> pd.DataFrame:
    """Build a merged DataFrame of adjusted close prices for all tickers.

    Parameters
    ----------
    raw_prices:
        Mapping ticker → raw API response dict.
    tickers:
        Ordered list of tickers (defines column order).

    Returns
    -------
    DataFrame with DatetimeIndex and one column per ticker.
    """
    series_list = []
    for t in tickers:
        s = monthly_prices_to_series(raw_prices[t])
        s.name = t
        series_list.append(s)
    prices = pd.concat(series_list, axis=1, join="inner").dropna()
    logger.info(
        "Aligned price data: %d monthly observations, tickers=%s",
        len(prices),
        tickers,
    )
    return prices


def preprocess(
    raw_prices: dict[str, dict],
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Full preprocessing pipeline.

    Returns
    -------
    mu : (N,) annualised mean log-return vector
    Sigma : (N, N) annualised covariance matrix
    tickers : list of ticker strings (matching mu/Sigma order)
    returns_df : monthly log-return DataFrame (for diagnostics)
    """
    prices = build_price_dataframe(raw_prices, tickers)
    log_ret_series = [compute_log_returns(prices[t]) for t in tickers]
    returns_df = align_returns(*log_ret_series)
    mu, Sigma = annualise_returns(returns_df)
    logger.info(
        "Annualised mu=%s  diag(Sigma)=%s", np.round(mu, 4), np.round(np.diag(Sigma), 4)
    )
    return mu, Sigma, tickers, returns_df
