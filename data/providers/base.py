"""
Abstract base class for market data providers.

All concrete providers (Kite, Upstox, …) must implement this interface so the
rest of the system can be swapped between brokers without code changes.

Canonical interval names (shared across all providers)::

    "minute"   – 1-minute bars
    "3minute"  – 3-minute bars
    "5minute"  – 5-minute bars
    "15minute" – 15-minute bars
    "30minute" – 30-minute bars
    "60minute" – 60-minute / hourly bars
    "day"      – daily bars

Each provider translates these into its own API-specific names internally.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Callable

import pandas as pd

# Intervals understood by the whole system.  Providers must support all of them.
CANONICAL_INTERVALS = frozenset(
    {"minute", "3minute", "5minute", "15minute", "30minute", "60minute", "day"}
)


class OHLCVProvider(ABC):
    """Abstract interface for fetching and streaming OHLCV market data."""

    # ------------------------------------------------------------------ #
    # Historical data                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        from_date: date | datetime,
        to_date: date | datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for *symbol* between *from_date* and *to_date*.

        The DataFrame returned must have at minimum these columns::

            time (index or column), open, high, low, close, volume, symbol, interval

        Implementors should also call ``data.store.write_ohlcv`` so results are
        persisted to TimescaleDB, mirroring the existing KiteIngestor contract.

        Args:
            symbol:    Instrument symbol, e.g. ``"RELIANCE"``.
            from_date: Start of the date range (inclusive).
            to_date:   End of the date range (inclusive).
            interval:  One of :data:`CANONICAL_INTERVALS`.

        Returns:
            DataFrame indexed by ``time``, or an empty DataFrame when no data
            is available.
        """

    # ------------------------------------------------------------------ #
    # Live streaming                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def stream_live(
        self,
        symbols: list[str],
        on_tick: Callable[[list[dict]], None] | None = None,
    ) -> None:
        """Start streaming live ticks for *symbols*.

        Blocks until :meth:`stop_stream` is called or the connection drops.

        Each tick dict passed to *on_tick* should contain at minimum::

            {"symbol": str, "last_price": float, "timestamp": str | datetime}

        Implementors should also write ticks to Redis via
        ``data.store.write_tick`` so the rest of the pipeline sees them.
        """

    @abstractmethod
    def stop_stream(self) -> None:
        """Gracefully stop the live stream."""

    # ------------------------------------------------------------------ #
    # Quotes                                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        """Return the latest market quote for each symbol.

        Keys are the requested symbol strings; values are provider-specific
        quote dicts (typically containing ``last_price``, ``ohlc``, etc.).
        """

    # ------------------------------------------------------------------ #
    # Auth helpers (optional override)                                     #
    # ------------------------------------------------------------------ #

    def get_login_url(self) -> str:
        """Return the OAuth / login URL for daily token refresh (if applicable)."""
        raise NotImplementedError(f"{type(self).__name__} does not require a login URL")

    def refresh_access_token(self, request_token: str) -> str:
        """Exchange a request token for an access token (if applicable)."""
        raise NotImplementedError(f"{type(self).__name__} does not support token refresh")
