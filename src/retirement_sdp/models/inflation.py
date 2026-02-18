"""Inflation model: CPI-based or fixed-rate real deflators.

Two modes:
1. **CPI mode** — uses downloaded CPI index to compute annual inflation rates,
   then averages over the historical window.
2. **Fixed mode** — uses the ``inflation_rate`` from config as a constant.

The output is a scalar annual inflation rate used to adjust cash flows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_annual_inflation_from_cpi(
    cpi_data: dict[str, str],
    min_years: int = 5,
) -> float:
    """Estimate annual inflation rate from CPI data.

    Parameters
    ----------
    cpi_data:
        Mapping ``"YYYY-MM-DD"`` → CPI value string (from Alpha Vantage).
    min_years:
        Minimum years of history required; if fewer observations, returns NaN.

    Returns
    -------
    Geometric mean annual CPI inflation rate, or NaN if data is insufficient.
    """
    if not cpi_data:
        return float("nan")

    records = {
        pd.Timestamp(d): float(v)
        for d, v in cpi_data.items()
        if v not in ("", ".", "None", None)
    }
    if not records:
        return float("nan")

    s = pd.Series(records).sort_index()
    annual = s.resample("YE").last().dropna()

    if len(annual) < min_years + 1:
        logger.warning(
            "Only %d annual CPI obs; need %d+1. Falling back to fixed rate.",
            len(annual),
            min_years,
        )
        return float("nan")

    annual_log_changes = np.log(annual / annual.shift(1)).dropna().values
    geometric_mean = float(np.exp(annual_log_changes.mean()) - 1)
    logger.info("CPI-derived annual inflation: %.4f", geometric_mean)
    return geometric_mean


class InflationModel:
    """Provides annual inflation rate for cash-flow deflation.

    Parameters
    ----------
    cpi_data:
        Raw CPI dict from AlphaVantageClient (may be empty).
    fallback_rate:
        Fixed annual rate used when CPI data is unavailable or insufficient.
    """

    def __init__(
        self,
        cpi_data: dict[str, str] | None = None,
        fallback_rate: float = 0.03,
    ) -> None:
        self._fallback = fallback_rate
        cpi_rate = compute_annual_inflation_from_cpi(cpi_data or {})
        if np.isnan(cpi_rate):
            self.annual_rate = fallback_rate
            self._source = "fixed"
        else:
            self.annual_rate = cpi_rate
            self._source = "CPI"
        logger.info(
            "InflationModel: rate=%.4f (source=%s)", self.annual_rate, self._source
        )

    def deflator(self, t: int) -> float:
        """Cumulative real deflator at year *t*: (1 + pi)^t."""
        return (1.0 + self.annual_rate) ** t

    def real_to_nominal(self, real_value: float, t: int) -> float:
        """Convert a real (base-year) value to nominal at year *t*."""
        return real_value * self.deflator(t)

    def __repr__(self) -> str:
        return f"InflationModel(rate={self.annual_rate:.4f}, source={self._source})"
