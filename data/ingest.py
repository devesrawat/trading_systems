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


# ---------------------------------------------------------------------------
# NSE market-data scraper (Phase 1 — M1 spec)
# ---------------------------------------------------------------------------

class NSEDataScraper:
    """
    Fetches NSE-specific data not available through broker APIs:
    FII/DII daily flows, India VIX, NSE corporate announcements.

    Uses NSE's undocumented JSON endpoints. A cookie-primed session is
    established on first call and reused for all subsequent requests.
    """

    BASE = "https://www.nseindia.com"
    _session: object | None = None

    def _get_session(self):  # type: ignore[return]
        import requests
        if self._session is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": self.BASE,
            })
            s.get(self.BASE, timeout=10)  # establish cookies
            self._session = s
        return self._session

    def get_fii_dii_flows(self) -> dict:
        """
        Returns latest FII/DII flow data.

        Keys: fii_net_cash, dii_net_cash, fii_net_fno (all INR crore), date
        """
        session = self._get_session()
        resp = session.get(f"{self.BASE}/api/fiidiiTradeReact", timeout=10)
        resp.raise_for_status()
        latest = resp.json()[0]
        return {
            "fii_net_cash": float(latest.get("netval_fii_cash", 0)),
            "dii_net_cash": float(latest.get("netval_dii_cash", 0)),
            "fii_net_fno": float(latest.get("netval_fii_fno", 0)),
            "date": latest.get("date", ""),
        }

    def get_india_vix(self) -> float:
        """Return current India VIX value."""
        session = self._get_session()
        resp = session.get(f"{self.BASE}/api/allIndices", timeout=10)
        resp.raise_for_status()
        for idx in resp.json().get("data", []):
            if idx.get("indexSymbol") == "INDIA VIX":
                return float(idx["last"])
        raise ValueError("India VIX not found in NSE API response")


# ---------------------------------------------------------------------------
# Crypto metrics collector (Phase 1 — M1 spec)
# ---------------------------------------------------------------------------

class CryptoMetricsCollector:
    """
    Collects on-chain and derivatives metrics for BTC/ETH/SOL.

    All endpoints are public / free-tier — no API key required.
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    def get_fear_greed(self) -> int:
        """Return 0-100 Fear & Greed index. 0 = extreme fear, 100 = extreme greed."""
        import requests
        resp = requests.get(self.FEAR_GREED_URL, timeout=5)
        return int(resp.json()["data"][0]["value"])

    def get_binance_funding_rate(self, symbol: str) -> float:
        """Return current 8-hour funding rate from Binance futures."""
        import requests
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": f"{symbol}USDT", "limit": 1},
            timeout=5,
        )
        data = resp.json()
        return float(data[0]["fundingRate"]) if data else 0.0

    def get_binance_open_interest(self, symbol: str) -> float:
        """Return open interest in USDT from Binance futures."""
        import requests
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": f"{symbol}USDT"},
            timeout=5,
        )
        return float(resp.json()["openInterest"])
