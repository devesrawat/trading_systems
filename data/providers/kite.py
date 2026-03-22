"""
Zerodha Kite Connect adapter implementing :class:`OHLCVProvider`.

This is a thin wrapper around :class:`data.ingest.KiteIngestor` so that Kite
can be used interchangeably with other providers via the common interface.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Callable

import pandas as pd
import structlog

from data.ingest import KiteIngestor
from data.store import get_redis

from .base import OHLCVProvider

log = structlog.get_logger(__name__)

# Canonical interval → Kite interval (they match 1-to-1 in this case)
_INTERVAL_MAP: dict[str, str] = {
    "minute": "minute",
    "3minute": "3minute",
    "5minute": "5minute",
    "15minute": "15minute",
    "30minute": "30minute",
    "60minute": "60minute",
    "day": "day",
}


class KiteProvider(OHLCVProvider):
    """Zerodha Kite Connect data provider.

    Before fetching or streaming, register the symbol → instrument-token
    mapping with :meth:`register_tokens`::

        provider = KiteProvider(api_key="...", access_token="...")
        provider.register_tokens({"RELIANCE": 738561, "INFY": 408065})
        df = provider.fetch_historical("RELIANCE", from_date, to_date, "day")
    """

    def __init__(self, api_key: str, access_token: str | None = None) -> None:
        self._ingestor = KiteIngestor(api_key=api_key, access_token=access_token)
        self._symbol_to_token: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Token registry                                                       #
    # ------------------------------------------------------------------ #

    def register_tokens(self, mapping: dict[str, int]) -> None:
        """Register *symbol → instrument token* pairs needed for API calls."""
        self._symbol_to_token.update(mapping)

    # ------------------------------------------------------------------ #
    # OHLCVProvider interface                                              #
    # ------------------------------------------------------------------ #

    def fetch_historical(
        self,
        symbol: str,
        from_date: date | datetime,
        to_date: date | datetime,
        interval: str,
    ) -> pd.DataFrame:
        token = self._symbol_to_token.get(symbol)
        if token is None:
            raise ValueError(
                f"No instrument token registered for '{symbol}'. "
                "Call register_tokens() first."
            )
        kite_interval = _INTERVAL_MAP.get(interval, interval)
        return self._ingestor.fetch_historical(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=kite_interval,
            symbol=symbol,
        )

    def stream_live(
        self,
        symbols: list[str],
        on_tick: Callable[[list[dict]], None] | None = None,
    ) -> None:
        missing = [s for s in symbols if s not in self._symbol_to_token]
        if missing:
            log.warning("kite_provider_missing_tokens", symbols=missing)
        tokens = [self._symbol_to_token[s] for s in symbols if s in self._symbol_to_token]
        self._ingestor.stream_live(tokens=tokens, on_tick_extra=on_tick)

    def stop_stream(self) -> None:
        self._ingestor.stop_stream()

    def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        trading_symbols = [f"NSE:{s}" for s in symbols]
        return self._ingestor.kite.quote(trading_symbols)

    # ------------------------------------------------------------------ #
    # Auth helpers                                                         #
    # ------------------------------------------------------------------ #

    def get_login_url(self) -> str:
        return self._ingestor.get_login_url()

    def refresh_access_token(self, request_token: str) -> str:
        return self._ingestor.refresh_access_token(request_token)

    # ------------------------------------------------------------------ #
    # Kite-specific extras                                                 #
    # ------------------------------------------------------------------ #

    @property
    def ingestor(self) -> KiteIngestor:
        """Direct access to the underlying :class:`KiteIngestor` instance."""
        return self._ingestor
