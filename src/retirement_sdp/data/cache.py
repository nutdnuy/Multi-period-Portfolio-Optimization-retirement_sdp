"""Parquet-based local cache for Alpha Vantage price data.

Avoids re-fetching data on every run (respects the 25 calls/day free tier).
Cache files are stored under *cache_dir* as ``<ticker>_monthly.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetCache:
    """Read / write pandas DataFrames to Parquet files.

    Parameters
    ----------
    cache_dir:
        Directory in which cache files are stored.  Created on first write.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(" ", "_")
        return self._dir / f"{safe}.parquet"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> pd.DataFrame:
        """Load cached DataFrame for *key*.

        Raises
        ------
        FileNotFoundError
            If the cache entry does not exist.
        """
        p = self._path(key)
        if not p.exists():
            raise FileNotFoundError(f"No cache entry for key '{key}' at {p}")
        logger.debug("Cache HIT: %s", p)
        return pd.read_parquet(p)

    def save(self, key: str, df: pd.DataFrame) -> None:
        """Persist *df* to Parquet under *key*."""
        p = self._path(key)
        df.to_parquet(p, index=True)
        logger.debug("Cache WRITE: %s  (%d rows)", p, len(df))

    def load_or_fetch(
        self,
        key: str,
        fetch_fn,  # callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """Return cached data if available, otherwise call *fetch_fn* and cache."""
        if self.exists(key):
            return self.load(key)
        logger.info("Cache MISS for '%s' — fetching from API.", key)
        df = fetch_fn()
        self.save(key, df)
        return df
