"""
Zerodha Kite data ingestor.

Responsibilities:
  - Fetch historical OHLCV bars and write to TimescaleDB
  - Stream live ticks via KiteTicker WebSocket and cache in Redis
  - Handle daily access-token refresh
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Callable

import pandas as pd
import structlog
from kiteconnect import KiteConnect, KiteTicker

from config.settings import settings
from data.providers.base import OHLCVProvider
from data.redis_keys import RedisKeys
from data.store import get_redis, write_ohlcv, write_tick

log = structlog.get_logger(__name__)

# Kite supported intervals
VALID_INTERVALS = frozenset(
    {"minute", "3minute", "5minute", "15minute", "30minute", "60minute", "day"}
)

# Max lookback for minute-level data per Kite API limit (60 days per request)
MINUTE_CHUNK_DAYS = 60


class KiteIngestor:
    """Wraps KiteConnect for historical fetch and live tick streaming."""

    def __init__(self, api_key: str, access_token: str | None = None) -> None:
        self._kite = KiteConnect(api_key=api_key)
        if access_token:
            self._kite.set_access_token(access_token)
        self._ticker: KiteTicker | None = None
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        instrument_token: int,
        from_date: date | datetime,
        to_date: date | datetime,
        interval: str,
        symbol: str = "",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for *instrument_token* between *from_date* and
        *to_date* at the given *interval*, writing results to TimescaleDB.

        Minute-level data is fetched in 60-day chunks to respect Kite limits.
        Returns the full concatenated DataFrame.
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Must be one of {VALID_INTERVALS}")

        chunks = OHLCVProvider._date_chunks(from_date, to_date, interval, MINUTE_CHUNK_DAYS)
        frames: list[pd.DataFrame] = []

        for chunk_from, chunk_to in chunks:
            log.info(
                "fetching_historical",
                token=instrument_token,
                interval=interval,
                from_date=chunk_from,
                to_date=chunk_to,
            )
            raw = self._kite.historical_data(
                instrument_token=instrument_token,
                from_date=chunk_from,
                to_date=chunk_to,
                interval=interval,
            )
            if not raw:
                continue

            df = pd.DataFrame(raw)
            df.rename(columns={"date": "time"}, inplace=True)
            df["token"] = instrument_token
            df["symbol"] = symbol
            df["interval"] = interval
            df.set_index("time", inplace=True)
            frames.append(df)

        if not frames:
            log.warning("no_historical_data", token=instrument_token, interval=interval)
            return pd.DataFrame()

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")]
        write_ohlcv(result.reset_index())
        log.info("historical_written", token=instrument_token, rows=len(result))
        return result

    # ------------------------------------------------------------------
    # Live streaming
    # ------------------------------------------------------------------

    def stream_live(
        self,
        tokens: list[int],
        on_tick_extra: Callable[[list[dict]], None] | None = None,
    ) -> None:
        """
        Start the KiteTicker WebSocket and stream live ticks to Redis.

        *on_tick_extra* is an optional callback for additional processing
        (e.g. feeding into the signal pipeline).

        This call blocks until the connection is closed or max retries exceeded.
        """
        api_key = self._api_key
        access_token = self._kite.access_token

        ticker = KiteTicker(api_key, access_token)
        self._ticker = ticker
        self._reconnect_count = 0

        def on_ticks(ws: KiteTicker, ticks: list[dict]) -> None:
            for tick in ticks:
                write_tick(tick["instrument_token"], tick)
            if on_tick_extra:
                on_tick_extra(ticks)

        def on_connect(ws: KiteTicker, response: dict) -> None:
            log.info("kite_ticker_connected", tokens=len(tokens))
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            self._reconnect_count = 0

        def on_error(ws: KiteTicker, code: int, reason: str) -> None:
            log.error("kite_ticker_error", code=code, reason=reason)

        def on_close(ws: KiteTicker, code: int, reason: str) -> None:
            log.warning("kite_ticker_closed", code=code, reason=reason)

        def on_reconnect(ws: KiteTicker, attempts_count: int) -> None:
            self._reconnect_count = attempts_count
            log.warning("kite_ticker_reconnecting", attempt=attempts_count)
            if attempts_count > 5:
                log.error("kite_ticker_max_retries_exceeded")
                ws.stop()

        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_error = on_error
        ticker.on_close = on_close
        ticker.on_reconnect = on_reconnect

        log.info("starting_kite_ticker", tokens=tokens)
        ticker.connect(threaded=False)

    def stop_stream(self) -> None:
        if self._ticker:
            self._ticker.stop()
            log.info("kite_ticker_stopped")

    # ------------------------------------------------------------------
    # Access token management
    # ------------------------------------------------------------------

    def get_login_url(self) -> str:
        """Return the Kite login URL for daily token refresh."""
        return self._kite.login_url()

    def refresh_access_token(self, request_token: str) -> str:
        """
        Exchange a request_token (from the login redirect) for an access token.
        Stores the new token on the KiteConnect instance.

        Returns the new access token.
        """
        session = self._kite.generate_session(request_token, api_secret=settings.kite_api_secret)
        access_token: str = session["access_token"]
        self._kite.set_access_token(access_token)
        log.info("access_token_refreshed")

        # Persist to Redis so other processes can pick it up
        r = get_redis()
        r.set(RedisKeys.KITE_ACCESS_TOKEN, access_token)
        return access_token

    @property
    def kite(self) -> KiteConnect:
        return self._kite
