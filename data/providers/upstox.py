"""
Upstox v2 market data provider implementing :class:`OHLCVProvider`.

Uses only standard-library-adjacent packages that are already present in
the environment (``requests`` via kiteconnect, ``websocket-client`` via
kiteconnect).  No extra SDK required.

Upstox API v2 reference: https://upstox.com/developer/api-documentation/

Auth flow (once per trading day)::

    provider = UpstoxProvider(api_key="...", access_token="")
    print(provider.get_login_url())   # open URL in browser, log in
    code = input("Paste 'code' param from redirect URL: ")
    token = provider.refresh_access_token(code)
    # Persist token to UPSTOX_ACCESS_TOKEN env var

Instrument keys::

    Upstox identifies instruments by a key of the form ``"EXCHANGE|ISIN"``,
    for example ``"NSE_EQ|INE002A01018"`` for Reliance.  Download the master
    list at:
        https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz

    Use ``register_instruments()`` to map friendly symbol names to these keys.

Live streaming notes::

    The Upstox v2 feed sends **protobuf-encoded** binary frames.  This module
    will attempt to decode them using the ``upstox_client`` package if it is
    installed (``pip install upstox-python`` from the Upstox GitHub release
    page).  If the package is absent, raw bytes are passed as-is to
    ``on_tick`` and ticks are not written to Redis — useful for debugging but
    not for production.  Historical data and quotes work with zero extra deps.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime
from typing import Callable
from urllib.parse import urlencode

import pandas as pd
import requests
import structlog

from data.store import write_ohlcv, write_tick

from .base import OHLCVProvider

log = structlog.get_logger(__name__)

_BASE_URL = "https://api.upstox.com/v2"
_WS_URL = "wss://api.upstox.com/v2/feed/market-data-feed"
_AUTH_DIALOG = "https://api.upstox.com/v2/login/authorization/dialog"
_TOKEN_URL = f"{_BASE_URL}/login/authorization/token"

# Canonical interval → Upstox REST interval string
_INTERVAL_MAP: dict[str, str] = {
    "minute": "1minute",
    "3minute": "3minute",
    "5minute": "5minute",
    "15minute": "15minute",
    "30minute": "30minute",
    "60minute": "60minute",
    "day": "1day",
}

# Max days per intraday historical request (Upstox limit ≈ 100 days)
_INTRADAY_CHUNK_DAYS = 100


class UpstoxProvider(OHLCVProvider):
    """Upstox v2 market data adapter.

    No extra packages required beyond what ``kiteconnect`` already pulls in.
    For live streaming protobuf decoding, optionally install the Upstox SDK
    from GitHub (see module docstring).

    Basic usage::

        provider = UpstoxProvider(api_key="...", access_token="<daily token>")
        provider.register_instruments({
            "RELIANCE": "NSE_EQ|INE002A01018",
            "INFY":     "NSE_EQ|INE009A01021",
        })
        df = provider.fetch_historical("RELIANCE", date(2024,1,1), date.today(), "day")
    """

    def __init__(
        self,
        api_key: str,
        access_token: str | None = None,
        redirect_uri: str = "http://localhost:8080",
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token or ""
        self._redirect_uri = redirect_uri
        self._ws_app = None
        self._ws_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # symbol → Upstox instrument key, e.g. "NSE_EQ|INE002A01018"
        self._symbol_to_key: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Instrument registry                                                  #
    # ------------------------------------------------------------------ #

    def register_instruments(self, mapping: dict[str, str]) -> None:
        """Register *symbol → Upstox instrument key* pairs.

        Download the master instrument list from::

            https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz

        The ``instrument_key`` column in that CSV is what goes here.
        """
        self._symbol_to_key.update(mapping)

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
        instrument_key = self._symbol_to_key.get(symbol)
        if not instrument_key:
            raise ValueError(
                f"No Upstox instrument key registered for '{symbol}'. "
                "Call register_instruments() first."
            )

        upstox_interval = _INTERVAL_MAP.get(interval, interval)
        chunks = self._date_chunks(from_date, to_date, interval, _INTRADAY_CHUNK_DAYS)
        frames: list[pd.DataFrame] = []

        for chunk_from, chunk_to in chunks:
            from_str = _to_date_str(chunk_from)
            to_str = _to_date_str(chunk_to)
            log.info(
                "upstox_fetching_historical",
                symbol=symbol,
                interval=upstox_interval,
                from_date=from_str,
                to_date=to_str,
            )

            # REST endpoint: GET /v2/historical-candle/{key}/{interval}/{to}/{from}
            url = (
                f"{_BASE_URL}/historical-candle"
                f"/{instrument_key}/{upstox_interval}/{to_str}/{from_str}"
            )
            resp = self._get(url)

            candles = resp.get("data", {}).get("candles", [])
            if not candles:
                continue

            # Each candle: [timestamp, open, high, low, close, volume, oi]
            df = pd.DataFrame(
                candles,
                columns=["time", "open", "high", "low", "close", "volume", "oi"],
            )
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata")
            df["symbol"] = symbol
            df["interval"] = interval
            df.set_index("time", inplace=True)
            frames.append(df)

        if not frames:
            log.warning("upstox_no_historical_data", symbol=symbol, interval=interval)
            return pd.DataFrame()

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")].sort_index()
        write_ohlcv(result.reset_index())
        log.info("upstox_historical_written", symbol=symbol, rows=len(result))
        return result

    def stream_live(
        self,
        symbols: list[str],
        on_tick: Callable[[list[dict]], None] | None = None,
    ) -> None:
        """Start the Upstox WebSocket feed (blocks until stopped).

        Protobuf decoding is attempted via the ``upstox_client`` package if
        available; otherwise raw bytes are forwarded to ``on_tick``.
        """
        import websocket  # websocket-client, already present via kiteconnect

        instrument_keys = [self._symbol_to_key[s] for s in symbols if s in self._symbol_to_key]
        missing = [s for s in symbols if s not in self._symbol_to_key]
        if missing:
            log.warning("upstox_stream_missing_instruments", symbols=missing)
        if not instrument_keys:
            log.warning("upstox_stream_no_instruments")
            return

        self._stop_event.clear()
        decoder = _try_load_protobuf_decoder()

        def on_open(ws: websocket.WebSocketApp) -> None:
            log.info("upstox_ws_connected", instruments=len(instrument_keys))
            subscribe_msg = json.dumps(
                {
                    "guid": f"sub-{int(time.time())}",
                    "method": "sub",
                    "data": {
                        "mode": "full",
                        "instrumentKeys": instrument_keys,
                    },
                }
            )
            ws.send(subscribe_msg)

        def on_message(ws: websocket.WebSocketApp, message: bytes) -> None:
            ticks: list[dict]
            if decoder is not None:
                ticks = decoder(message)
            else:
                # Protobuf decoder unavailable — pass raw bytes for debugging
                ticks = [{"raw": message}]
                log.debug("upstox_ws_raw_tick_no_decoder")

            for tick in ticks:
                symbol_key = tick.get("symbol", "")
                if symbol_key:
                    write_tick(symbol_key, tick)
            if on_tick and ticks:
                on_tick(ticks)

        def on_error(ws: websocket.WebSocketApp, error: Exception) -> None:
            log.error("upstox_ws_error", error=str(error))

        def on_close(ws: websocket.WebSocketApp, code: int, reason: str) -> None:
            log.warning("upstox_ws_closed", code=code, reason=reason)

        ws_app = websocket.WebSocketApp(
            _WS_URL,
            header={"Authorization": f"Bearer {self._access_token}"},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws_app = ws_app
        log.info("starting_upstox_ws", instruments=instrument_keys)
        ws_app.run_forever(reconnect=5)

    def stop_stream(self) -> None:
        if self._ws_app:
            self._ws_app.close()
            self._stop_event.set()
            log.info("upstox_ws_stopped")

    def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        keys = [self._symbol_to_key[s] for s in symbols if s in self._symbol_to_key]
        if not keys:
            return {}

        resp = self._get(
            f"{_BASE_URL}/market-quote/quotes",
            params={"symbol": ",".join(keys)},
        )
        raw: dict = resp.get("data", {})

        # Build symbol-keyed result
        key_to_symbol = {v: k for k, v in self._symbol_to_key.items() if k in symbols}
        return {
            key_to_symbol[key]: quote
            for key, quote in raw.items()
            if key in key_to_symbol
        }

    # ------------------------------------------------------------------ #
    # Auth helpers                                                         #
    # ------------------------------------------------------------------ #

    def get_login_url(self) -> str:
        """Return the Upstox OAuth2 URL.  Open in a browser, log in, then
        copy the ``code`` query param from the redirect URL and pass it to
        :meth:`refresh_access_token`."""
        params = {
            "response_type": "code",
            "client_id": self._api_key,
            "redirect_uri": self._redirect_uri,
        }
        return f"{_AUTH_DIALOG}?{urlencode(params)}"

    def refresh_access_token(self, code: str) -> str:
        """Exchange an auth *code* for an access token.  Returns the token."""
        resp = requests.post(
            _TOKEN_URL,
            data={
                "code": code,
                "client_id": self._api_key,
                "redirect_uri": self._redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        token: str = resp.json()["access_token"]
        self._access_token = token
        log.info("upstox_access_token_refreshed")
        return token

    # ------------------------------------------------------------------ #
    # Internal HTTP helper                                                 #
    # ------------------------------------------------------------------ #

    def _get(self, url: str, params: dict | None = None) -> dict:
        resp = requests.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------ #
# Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _to_date_str(d: date | datetime) -> str:
    return (d.date() if isinstance(d, datetime) else d).isoformat()


def _try_load_protobuf_decoder() -> Callable[[bytes], list[dict]] | None:
    """Try to load the upstox_client protobuf decoder.

    Returns a callable that decodes raw WebSocket bytes into a list of tick
    dicts, or ``None`` if the SDK is not installed.

    Install the SDK (GitHub release, not PyPI) to enable full decoding::

        pip install git+https://github.com/upstox/upstox-python.git
    """
    try:
        from upstox_client.feeder.upstox_pb2 import FeedResponse  # type: ignore[import]

        def decode(data: bytes) -> list[dict]:
            feed = FeedResponse()
            feed.ParseFromString(data)
            ticks: list[dict] = []
            for key, feed_data in feed.feeds.items():
                ff = feed_data.ff
                market_ff = ff.marketFF if ff.HasField("marketFF") else ff.indexFF
                ltpc = market_ff.ltpc
                ticks.append(
                    {
                        "symbol": key,
                        "last_price": ltpc.ltp,
                        "last_quantity": ltpc.ltq,
                        "close_price": ltpc.cp,
                        "timestamp": ltpc.ltt,
                    }
                )
            return ticks

        log.info("upstox_protobuf_decoder_loaded")
        return decode

    except ImportError:
        log.warning(
            "upstox_protobuf_decoder_unavailable",
            hint="pip install git+https://github.com/upstox/upstox-python.git",
        )
        return None
