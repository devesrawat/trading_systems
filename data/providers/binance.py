"""
Binance public market data adapter implementing :class:`OHLCVProvider`.

No API key is required for market data (all endpoints used here are public).
Optional ``api_key`` / ``api_secret`` are accepted for future order-placement
extensions but are not used by this module.

Canonical interval mapping::

    minute   → 1m    3minute → 3m    5minute  → 5m
    15minute → 15m   30minute→ 30m   60minute → 1h    day → 1d

Symbol convention::

    Binance uses trading-pair strings, e.g. ``"BTCUSDT"``, ``"ETHUSDT"``.
    ``register_instruments()`` accepts a ``dict[str, str]`` mapping a friendly
    name to the Binance pair::

        provider.register_instruments({"BTC": "BTCUSDT", "ETH": "ETHUSDT"})

    If a friendly name equals the Binance pair you can also pass::

        provider.register_instruments({"BTCUSDT": "BTCUSDT"})

Free-tier rate limits (no auth):
    REST:      1 200 request-weight / min.  ``GET /klines`` costs weight 2.
    WebSocket: up to 1 024 streams per connection.

Live streaming uses Binance aggregated-trade streams (``@aggTrade``), which
fire on every market trade — closest to the NSE tick semantics used elsewhere.
Each tick is written to Redis via :func:`data.store.write_crypto_tick`.

Database storage reuses the existing ``ohlcv`` hypertable.  Because Binance
has no integer instrument tokens, a deterministic 32-bit hash of the pair
string is used in the ``token`` column so the primary-key constraint is met.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, date, datetime
from typing import Any, Callable

import pandas as pd
import requests
import structlog

from data.store import write_crypto_tick, write_ohlcv

from .base import CANONICAL_INTERVALS, OHLCVProvider

log = structlog.get_logger(__name__)

_REST_BASE = "https://api.binance.com/api/v3"
_WS_BASE = "wss://stream.binance.com:9443/stream"

# Canonical interval → Binance REST/WebSocket interval string
_INTERVAL_MAP: dict[str, str] = {
    "minute": "1m",
    "3minute": "3m",
    "5minute": "5m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "1h",
    "day": "1d",
}

# Binance returns at most 1 000 candles per klines request
_MAX_BARS_PER_REQUEST = 1000

# Weight for a single /klines call (tracked for self-imposed rate cap)
_KLINES_WEIGHT = 2

# Seconds to sleep when we approach the weight limit (conservative)
_WEIGHT_SLEEP = 0.12  # ~8 calls/s → ~16 weight/s → well under 1 200/min


def _symbol_hash(pair: str) -> int:
    """Derive a stable 32-bit unsigned integer from *pair* for the DB token column.

    Uses the first 8 hex digits of the MD5 digest — deterministic, collision-free
    for any realistic crypto universe, and stays positive.
    """
    return int(hashlib.md5(pair.upper().encode(), usedforsecurity=False).hexdigest()[:8], 16)


def _to_ms(dt: date | datetime) -> int:
    """Convert a :class:`date` or :class:`datetime` to a millisecond timestamp."""
    if isinstance(dt, datetime):
        ts = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
        return int(ts.timestamp() * 1000)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=UTC).timestamp() * 1000)


class BinanceProvider(OHLCVProvider):
    """Binance public REST + WebSocket market data provider.

    No authentication is required.  Just instantiate and call
    :meth:`register_instruments` with the pairs you want::

        provider = BinanceProvider()
        provider.register_instruments({"BTC": "BTCUSDT", "ETH": "ETHUSDT"})
        df = provider.fetch_historical("BTC", date(2024, 1, 1), date.today(), "day")
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        # api_key / api_secret reserved for future order-placement extension
        self._api_key = api_key or ""
        self._api_secret = api_secret or ""

        # friendly_name → Binance pair (e.g. "BTC" → "BTCUSDT")
        self._name_to_pair: dict[str, str] = {}
        # Binance pair → friendly name for reverse lookup in get_quote
        # ("BTCUSDT" → "BTC").  Maintained separately so the auto-alias
        # "BTCUSDT" → "BTCUSDT" does not overwrite the friendly mapping.
        self._pair_to_friendly: dict[str, str] = {}

        self._ws_app = None
        self._ws_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Instrument registry                                                  #
    # ------------------------------------------------------------------ #

    def register_instruments(self, mapping: dict[str, str]) -> None:  # type: ignore[override]
        """Register *friendly_name → Binance pair* mappings.

        Example::

            provider.register_instruments(
                {
                    "BTC": "BTCUSDT",
                    "ETH": "ETHUSDT",
                    "SOL": "SOLUSDT",
                }
            )

        Both the friendly name and the raw pair (``"BTCUSDT"``) can be used
        interchangeably in subsequent calls once registered.
        """
        for name, pair in mapping.items():
            pair_upper = pair.upper()
            name_upper = name.upper()
            self._name_to_pair[name_upper] = pair_upper
            # Also register the pair itself so callers can pass either form
            self._name_to_pair[pair_upper] = pair_upper
            # Reverse map: prefer the friendly name; fall back to the pair itself
            self._pair_to_friendly.setdefault(pair_upper, name_upper)

    def _resolve_pair(self, symbol: str) -> str:
        """Translate *symbol* to a Binance trading pair, raising if unknown."""
        pair = self._name_to_pair.get(symbol.upper())
        if pair is None:
            raise ValueError(
                f"No Binance pair registered for '{symbol}'. Call register_instruments() first."
            )
        return pair

    # ------------------------------------------------------------------ #
    # Historical data                                                      #
    # ------------------------------------------------------------------ #

    def fetch_historical(
        self,
        symbol: str,
        from_date: date | datetime,
        to_date: date | datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Binance and persist to TimescaleDB.

        Paginates automatically — Binance caps each response at 1 000 bars.
        """
        if interval not in CANONICAL_INTERVALS:
            raise ValueError(f"Unknown interval '{interval}'. Use one of {CANONICAL_INTERVALS}.")

        pair = self._resolve_pair(symbol)
        binance_interval = _INTERVAL_MAP[interval]
        token = _symbol_hash(pair)

        start_ms = _to_ms(from_date)
        end_ms = _to_ms(to_date)

        frames: list[pd.DataFrame] = []
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            log.info(
                "binance_fetching_klines",
                pair=pair,
                interval=binance_interval,
                cursor=cursor_ms,
            )
            params: dict[str, Any] = {
                "symbol": pair,
                "interval": binance_interval,
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": _MAX_BARS_PER_REQUEST,
            }
            raw: list[list] = self._get("/klines", params)
            if not raw:
                break

            df = _klines_to_df(raw, symbol, pair, interval, token)
            frames.append(df)

            # Advance cursor past the last candle's close time
            last_close_ms: int = raw[-1][6]  # index 6 = close_time
            cursor_ms = last_close_ms + 1

            if len(raw) < _MAX_BARS_PER_REQUEST:
                break  # Binance returned fewer than max → we have all the data

            time.sleep(_WEIGHT_SLEEP)  # polite rate-limiting between pages

        if not frames:
            log.warning("binance_no_historical_data", symbol=symbol, interval=interval)
            return pd.DataFrame()

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")].sort_index()
        write_ohlcv(result.reset_index())
        log.info("binance_historical_written", symbol=symbol, rows=len(result))
        return result

    # ------------------------------------------------------------------ #
    # Live streaming (aggTrade)                                            #
    # ------------------------------------------------------------------ #

    def stream_live(
        self,
        symbols: list[str],
        on_tick: Callable[[list[dict]], None] | None = None,
    ) -> None:
        """Stream live aggregated-trade ticks for *symbols* via Binance WebSocket.

        Each tick written to Redis has the shape::

            {
                "symbol": "BTCUSDT",
                "last_price": float,
                "quantity": float,
                "timestamp": int,  # ms since epoch
                "is_buyer_maker": bool,
            }

        Blocks until :meth:`stop_stream` is called or the connection drops.
        Reconnects automatically every 5 s on error (handled by ``websocket``
        library's ``reconnect`` parameter).
        """
        import websocket  # websocket-client — already in deps via kiteconnect

        pairs = [self._resolve_pair(s) for s in symbols]
        streams = "/".join(f"{p.lower()}@aggTrade" for p in pairs)
        ws_url = f"{_WS_BASE}?streams={streams}"

        self._stop_event.clear()

        def on_open(ws: websocket.WebSocketApp) -> None:
            log.info("binance_ws_connected", pairs=pairs)

        def on_message(ws: websocket.WebSocketApp, message: str) -> None:
            try:
                envelope: dict = json.loads(message)
                data: dict = envelope.get("data", {})
                if data.get("e") != "aggTrade":
                    return

                tick: dict[str, Any] = {
                    "symbol": data["s"],
                    "last_price": float(data["p"]),
                    "quantity": float(data["q"]),
                    "timestamp": int(data["T"]),
                    "is_buyer_maker": bool(data["m"]),
                }
                write_crypto_tick(data["s"], tick)
                if on_tick:
                    on_tick([tick])
            except Exception as exc:
                log.error("binance_ws_parse_error", error=str(exc))

        def on_error(ws: websocket.WebSocketApp, error: Exception) -> None:
            log.error("binance_ws_error", error=str(error))

        def on_close(ws: websocket.WebSocketApp, code: int, reason: str) -> None:
            log.warning("binance_ws_closed", code=code, reason=reason)

        ws_app = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws_app = ws_app
        log.info("binance_ws_starting", url=ws_url)
        ws_app.run_forever(reconnect=5)

    def stop_stream(self) -> None:
        """Gracefully close the Binance WebSocket stream."""
        if self._ws_app:
            self._ws_app.close()
            self._stop_event.set()
            log.info("binance_ws_stopped")

    # ------------------------------------------------------------------ #
    # Quotes (REST snapshot)                                               #
    # ------------------------------------------------------------------ #

    def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        """Return the latest price for each symbol via REST.

        Uses ``/api/v3/ticker/price`` with a JSON array of pairs for a
        single request (request-weight 4 regardless of the number of pairs).
        """
        pairs = [self._resolve_pair(s) for s in symbols]
        pair_json = json.dumps(pairs)
        raw: list[dict] = self._get("/ticker/price", {"symbols": pair_json})

        result: dict[str, dict] = {}
        for item in raw:
            pair = item["symbol"]
            name = self._pair_to_friendly.get(pair, pair)
            result[name] = {
                "symbol": name,
                "pair": pair,
                "last_price": float(item["price"]),
            }
        return result

    # ------------------------------------------------------------------ #
    # Auth helpers (not applicable — public data needs no auth)           #
    # ------------------------------------------------------------------ #

    def get_login_url(self) -> str:
        raise NotImplementedError("Binance public endpoints require no login URL.")

    def refresh_access_token(self, request_token: str) -> str:
        raise NotImplementedError("Binance public endpoints require no token refresh.")

    # ------------------------------------------------------------------ #
    # Internal HTTP helper                                                 #
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``{_REST_BASE}{path}`` and return the parsed JSON body."""
        url = f"{_REST_BASE}{path}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _klines_to_df(
    raw: list[list],
    symbol: str,
    pair: str,
    interval: str,
    token: int,
) -> pd.DataFrame:
    """Convert a Binance ``/klines`` response list to a tidy DataFrame.

    Binance kline columns (by index):
        0  open_time  (ms)
        1  open
        2  high
        3  low
        4  close
        5  volume
        6  close_time (ms)
        7  quote_asset_volume
        8  number_of_trades
        9  taker_buy_base_asset_volume
        10 taker_buy_quote_asset_volume
        11 ignore
    """
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["token"] = token
    df["symbol"] = symbol
    df["interval"] = interval
    df = df.set_index("time")[
        ["token", "symbol", "open", "high", "low", "close", "volume", "interval"]
    ]
    return df
