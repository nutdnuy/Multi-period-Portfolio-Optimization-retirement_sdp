"""Alpha Vantage REST client with rate-limit retry."""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class AlphaVantageError(RuntimeError):
    pass


class AlphaVantageClient:
    """Thin wrapper around the Alpha Vantage REST API.

    Handles:
    - Rate-limit retries (5 attempts, exponential back-off)
    - Graceful error propagation
    - Monthly adjusted prices (adjusted close, split-adjusted)
    - CPI economic indicator series
    """

    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query") -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Core request method
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict:
        params = {"apikey": self._api_key, **params}
        resp = self._session.get(self._base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage returns HTTP 200 even for errors / rate-limits
        if "Note" in data:
            raise requests.exceptions.RequestException(
                f"Alpha Vantage rate-limit note: {data['Note']}"
            )
        if "Information" in data:
            raise AlphaVantageError(
                f"Alpha Vantage information message: {data['Information']}"
            )
        if "Error Message" in data:
            raise AlphaVantageError(f"Alpha Vantage error: {data['Error Message']}")
        return data

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_monthly_adjusted(self, ticker: str) -> dict[str, dict]:
        """Return TIME_SERIES_MONTHLY_ADJUSTED data for *ticker*.

        Returns
        -------
        dict mapping ``"YYYY-MM-DD"`` → dict with keys:
        ``"1. open"``, ``"2. high"``, ``"3. low"``,
        ``"4. close"``, ``"5. adjusted close"``, ``"6. volume"``,
        ``"7. dividend amount"``
        """
        logger.info("Fetching monthly adjusted prices for %s", ticker)
        data = self._get(
            {"function": "TIME_SERIES_MONTHLY_ADJUSTED", "symbol": ticker}
        )
        key = "Monthly Adjusted Time Series"
        if key not in data:
            raise AlphaVantageError(
                f"Unexpected response for {ticker}. Keys: {list(data.keys())}"
            )
        return data[key]

    def get_cpi(self, series_id: str = "CPIAUCSL") -> dict[str, str]:
        """Return CPI (or other economic indicator) as date → value mapping.

        Uses Alpha Vantage ECONOMIC_INDICATOR function.
        Falls back gracefully if the series is unavailable.
        """
        logger.info("Fetching economic indicator: %s", series_id)
        try:
            data = self._get(
                {
                    "function": "ECONOMIC_INDICATOR",
                    "symbol": series_id,
                    "interval": "monthly",
                }
            )
            if "data" not in data:
                raise AlphaVantageError(f"No 'data' key in response for {series_id}")
            return {row["date"]: row["value"] for row in data["data"]}
        except AlphaVantageError as exc:
            logger.warning(
                "CPI fetch failed (%s) — caller should fall back to fixed rate.", exc
            )
            return {}
