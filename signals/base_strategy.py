"""
BaseStrategy — contract every scan strategy must satisfy.

Rules for strategy authors
--------------------------
1. Set class-level `name`, `lookback_days`, `interval`.
2. Override `scan(symbol, df)` — return a dict on pass, None on fail.
3. `prepare()` runs the standard clean pipeline automatically.
   Override it only if you need non-standard cleaning (e.g. intraday gaps).
4. Do NOT call the DB, Redis, or any I/O inside `scan()` — it runs in a
   worker process and must be pure CPU.
5. All fields in the returned dict must be JSON-serialisable.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import structlog

from data.clean import (
    fill_missing_bars,
    flag_circuit_limit_days,
    remove_outliers,
    validate_ohlcv,
)

log = structlog.get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base for all scan strategies.

    Class attributes
    ----------------
    name          : unique snake_case identifier, used as the result dict key
    lookback_days : how many calendar days of history are required
    interval      : Kite interval string ('day', '5minute', etc.)
    min_bars      : minimum clean bars required before scan() is called
    """

    name: str          = "base"
    lookback_days: int = 400
    interval: str      = "day"
    min_bars: int      = 100

    # ------------------------------------------------------------------
    # Clean pipeline  (shared by all strategies, override if needed)
    # ------------------------------------------------------------------

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Validate and clean raw OHLCV records.

        Called by the engine before scan(). Returns None when data is
        too dirty or too short to be useful.
        """
        is_valid, issues = validate_ohlcv(df)
        if not is_valid:
            return None

        df = df.set_index("time")
        df = remove_outliers(df, col="close", method="zscore", threshold=4.0)
        df = remove_outliers(df, col="volume", method="zscore", threshold=4.0)

        if self.interval == "day":
            df = fill_missing_bars(df, interval="day")
            df = flag_circuit_limit_days(df)
            df = df[~df["circuit_hit"]].drop(columns=["circuit_hit"])

        if "is_filled" in df.columns:
            df = df.drop(columns=["is_filled"])

        if len(df) < self.min_bars:
            return None

        return df

    # ------------------------------------------------------------------
    # Strategy logic  (must be implemented by every subclass)
    # ------------------------------------------------------------------

    @abstractmethod
    def scan(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        """
        Evaluate *df* for the setup.

        Parameters
        ----------
        symbol : NSE trading symbol
        df     : clean OHLCV DataFrame with DatetimeIndex, columns
                 open / high / low / close / volume

        Returns
        -------
        A JSON-serialisable dict with at minimum::

            {"symbol": symbol, "strategy": self.name, ...setup-specific fields...}

        Return None if the symbol does not qualify.
        """

    # ------------------------------------------------------------------
    # Pickling support for ProcessPoolExecutor
    # ------------------------------------------------------------------

    @classmethod
    def fqname(cls) -> str:
        """Fully-qualified class name used by the engine to re-import in workers."""
        return f"{cls.__module__}.{cls.__qualname__}"

    def sort_key(self, result: dict[str, Any]) -> Any:
        """
        Key function used to order passing results.
        Default: alphabetical by symbol.  Override for strategy-specific ordering
        (e.g. sort VCP by distance_to_pivot_pct, RS by rs_score descending).
        """
        return result.get("symbol", "")

    @staticmethod
    def from_fqname(fqname: str) -> "BaseStrategy":
        """Instantiate a strategy from its fully-qualified name."""
        module_name, class_name = fqname.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls()
