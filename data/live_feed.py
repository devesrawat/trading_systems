"""
Real-time tick feed for 500 instruments via Kite WebSocket.

Architecture
------------
KiteTicker (WebSocket thread)
    │  ultra-thin on_ticks: push to deque, never block
    ▼
TickBuffer (deque, bounded at 20 000 ticks)
    │  background thread drains in batches
    ▼
TickProcessor
    ├── Redis pipeline  → flush latest tick per token in one round-trip
    ├── BarAggregator   → build 5-min OHLCV bars in memory per symbol
    │       └── on bar close → write_ohlcv() + publish "bars:5min" channel
    └── DayBarTracker   → update live day bar from MODE_QUOTE data
            └── publish "bars:day" channel on every day-bar update

Pub/Sub channels (consumers subscribe via Redis)
    bars:5min     — JSON: completed 5-min bar  {symbol, time, o, h, l, c, v}
    bars:day      — JSON: live day bar update   {symbol, time, o, h, l, c, v}
    breakout:vcp  — JSON: VCP breakout alert    {symbol, price, pivot, volume_ratio}

Tick modes
    MODE_LTP    for background universe (ticks ~7 bytes each)
    MODE_QUOTE  for VCP watchlist (includes live OHLC for the day, ~57 bytes)

Usage
-----
    feed = LiveFeed(api_key, access_token)
    feed.set_watchlist(vcp_symbols, mode="quote")
    feed.start(all_tokens)      # non-blocking, runs in background thread
    # subscribe to breakouts:
    feed.subscribe_breakouts(callback=lambda msg: print(msg))
    feed.stop()
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any, Callable

import pandas as pd
import structlog
from kiteconnect import KiteTicker

from data.redis_keys import RedisKeys
from data.store import get_redis, write_ohlcv

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICK_BUFFER_MAXLEN = 20_000  # drop oldest if consumer falls behind
_FLUSH_INTERVAL_SEC = 0.25  # drain buffer 4×/second
_BAR_INTERVAL_MIN = 5  # aggregate into 5-minute bars
_TICK_TTL_SEC = 10  # Redis TTL for latest-tick keys
_PUBSUB_BARS_5MIN = "bars:5min"
_PUBSUB_BARS_DAY = "bars:day"
_PUBSUB_BREAKOUT_VCP = "breakout:vcp"

# volume surge threshold for VCP breakout confirmation
_VOLUME_SURGE_RATIO = 1.40  # 40 % above 50-day average


# ---------------------------------------------------------------------------
# Bar aggregator — in-memory 5-min OHLCV builder
# ---------------------------------------------------------------------------


class _Bar:
    __slots__ = ("close", "high", "low", "open", "ts", "volume")

    def __init__(self, price: float, volume: int, ts: datetime) -> None:
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume
        self.ts = ts  # bar open timestamp (floored to interval)

    def update(self, price: float, volume: int) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += volume

    def to_dict(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "time": self.ts.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


def _floor_to_interval(dt: datetime, minutes: int) -> datetime:
    """Floor *dt* to the nearest *minutes*-minute boundary."""
    floored_min = (dt.minute // minutes) * minutes
    return dt.replace(minute=floored_min, second=0, microsecond=0)


class BarAggregator:
    """
    Maintains a rolling 5-min bar per symbol.
    Thread-safe via a per-symbol lock.
    """

    def __init__(self, interval_min: int = _BAR_INTERVAL_MIN) -> None:
        self._interval = interval_min
        self._bars: dict[str, _Bar] = {}
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._on_close: list[Callable[[str, _Bar], None]] = []

    def on_bar_close(self, callback: Callable[[str, _Bar], None]) -> None:
        """Register a callback fired when a bar is completed."""
        self._on_close.append(callback)

    def update(self, symbol: str, price: float, volume: int, ts: datetime) -> None:
        bar_ts = _floor_to_interval(ts, self._interval)

        with self._locks[symbol]:
            existing = self._bars.get(symbol)

            if existing is None:
                self._bars[symbol] = _Bar(price, volume, bar_ts)
                return

            if bar_ts > existing.ts:
                # bar boundary crossed — close the old bar, open a new one
                closed = existing
                self._bars[symbol] = _Bar(price, volume, bar_ts)
            else:
                existing.update(price, volume)
                return

        # fire callbacks outside the lock
        for cb in self._on_close:
            try:
                cb(symbol, closed)
            except Exception as exc:
                log.error("bar_close_callback_error", symbol=symbol, error=str(exc))


# ---------------------------------------------------------------------------
# VCP breakout checker — runs per-symbol on each bar close
# ---------------------------------------------------------------------------


class VCPBreakoutChecker:
    """
    Checks whether a completed bar breaks above the pre-computed pivot
    buy point with a volume surge.

    Pivot buy points are loaded from Redis at startup and refreshed
    whenever the daily VCP scanner re-runs.

    Redis key:  vcp:pivot:{symbol}  →  JSON {"pivot": 2895.5, "avg_vol_50d": 1234567}
    """

    def __init__(self) -> None:
        self._r = get_redis()

    def check(self, symbol: str, bar: _Bar) -> dict[str, Any] | None:
        """
        Returns a breakout dict if the bar qualifies, else None.
        """
        raw = self._r.get(f"vcp:pivot:{symbol}")
        if raw is None:
            return None  # symbol not on VCP watchlist

        meta = json.loads(raw)
        pivot = float(meta.get("pivot", 0))
        avg_vol = float(meta.get("avg_vol_50d", 1))

        if bar.close <= pivot:
            return None  # did not break above pivot

        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < _VOLUME_SURGE_RATIO:
            return None  # no volume confirmation

        return {
            "symbol": symbol,
            "price": bar.close,
            "pivot": pivot,
            "volume_ratio": round(vol_ratio, 2),
            "bar_time": bar.ts.isoformat(),
        }

    def store_pivot(self, symbol: str, pivot: float, avg_vol_50d: float) -> None:
        """Write/update a VCP pivot into Redis (called by the daily scanner)."""
        self._r.set(
            f"vcp:pivot:{symbol}",
            json.dumps({"pivot": pivot, "avg_vol_50d": avg_vol_50d}),
        )

    def clear_pivot(self, symbol: str) -> None:
        self._r.delete(f"vcp:pivot:{symbol}")


# ---------------------------------------------------------------------------
# Tick processor — drains the buffer in batches
# ---------------------------------------------------------------------------


class _TickProcessor:
    """
    Runs in a background daemon thread.
    Drains *buffer*, batches Redis writes, feeds BarAggregator.
    """

    def __init__(
        self,
        buffer: deque,
        token_to_symbol: dict[int, str],
        bar_agg: BarAggregator,
        vcp_checker: VCPBreakoutChecker,
    ) -> None:
        self._buf = buffer
        self._token_to_sym = token_to_symbol
        self._bar_agg = bar_agg
        self._vcp_checker = vcp_checker
        self._r = get_redis()
        self._running = False

    def start(self) -> None:
        self._running = True
        t = threading.Thread(target=self._run, daemon=True, name="tick-processor")
        t.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            if not self._buf:
                time.sleep(_FLUSH_INTERVAL_SEC)
                continue

            # Drain a batch — up to 500 ticks at a time
            batch: list[dict] = []
            try:
                while self._buf and len(batch) < 500:
                    batch.append(self._buf.popleft())
            except IndexError:
                pass

            if batch:
                self._process_batch(batch)

    def _process_batch(self, ticks: list[dict]) -> None:
        pipe = self._r.pipeline(transaction=False)

        for tick in ticks:
            token = tick.get("instrument_token")
            ltp = tick.get("last_price") or tick.get("last_traded_price")
            volume = int(tick.get("volume_traded") or tick.get("volume") or 0)
            ts_raw = tick.get("exchange_timestamp") or tick.get("timestamp")

            if not token or not ltp:
                continue

            symbol = self._token_to_sym.get(token)
            if not symbol:
                continue

            # Batch Redis tick update (one pipeline, not N round-trips)
            pipe.setex(RedisKeys.tick(token), _TICK_TTL_SEC, json.dumps(tick))

            # Update live day bar from MODE_QUOTE fields if present
            if "ohlc" in tick:
                self._update_day_bar(symbol, tick, pipe)

            # Aggregate into 5-min bar
            if ts_raw:
                ts = ts_raw if isinstance(ts_raw, datetime) else datetime.fromisoformat(str(ts_raw))
                self._bar_agg.update(symbol, float(ltp), volume, ts)

        # Single round-trip for all tick updates in this batch
        try:
            pipe.execute()
        except Exception as exc:
            log.error("redis_pipeline_flush_failed", error=str(exc))

    def _update_day_bar(self, symbol: str, tick: dict, pipe: Any) -> None:
        """Store live day-bar from MODE_QUOTE ohlc field."""
        ohlc = tick["ohlc"]
        bar = {
            "symbol": symbol,
            "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M"),
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": tick.get("last_price"),
            "volume": tick.get("volume_traded"),
        }
        pipe.set(f"bar:day:{symbol}", json.dumps(bar))
        pipe.publish(_PUBSUB_BARS_DAY, json.dumps(bar))


# ---------------------------------------------------------------------------
# LiveFeed — public API
# ---------------------------------------------------------------------------


class LiveFeed:
    """
    Manages a single KiteTicker WebSocket for up to 3000 tokens.

    Typical setup
    -------------
    1. Run the daily VCP scanner → get candidates + pivot levels
    2. Call feed.store_vcp_pivots(candidates) to register breakout levels
    3. Call feed.start(all_tokens, watchlist_tokens)
    4. Breakout alerts arrive via feed.subscribe_breakouts(callback)
    5. Call feed.stop() on shutdown
    """

    def __init__(self, api_key: str, access_token: str) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._ticker: KiteTicker | None = None
        self._token_map: dict[int, str] = {}  # token → symbol
        self._buffer: deque = deque(maxlen=_TICK_BUFFER_MAXLEN)
        self._bar_agg = BarAggregator()
        self._vcp_checker = VCPBreakoutChecker()
        self._processor: _TickProcessor | None = None
        self._breakout_cbs: list[Callable[[dict], None]] = []
        self._r = get_redis()

        # Wire bar-close events
        self._bar_agg.on_bar_close(self._on_bar_close)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_tokens(self, instruments: list[dict]) -> None:
        """
        Build the token → symbol map from instrument dicts.
        Call before start().
        """
        self._token_map = {i["instrument_token"]: i["tradingsymbol"] for i in instruments}

    def store_vcp_pivots(self, candidates: list[dict]) -> None:
        """
        Persist VCP pivot levels for breakout detection.
        *candidates* is the output of scan_vcp_universe().

        Requires avg_vol_50d in each candidate dict — add it from the DB
        before calling this.  Symbols NOT in candidates have their pivot cleared.
        """
        current_keys = set(self._r.keys("vcp:pivot:*"))
        new_symbols = set()

        for c in candidates:
            sym = c["symbol"]
            pivot = c["pivot_buy"]
            avg_vol = c.get("avg_vol_50d", 1_000_000)
            self._vcp_checker.store_pivot(sym, pivot, avg_vol)
            new_symbols.add(f"vcp:pivot:{sym}")

        # Remove stale pivots (stocks that dropped off the scan)
        for stale_key in current_keys - new_symbols:
            self._r.delete(stale_key)
            log.debug("vcp_pivot_cleared", key=stale_key)

        log.info("vcp_pivots_stored", count=len(candidates))

    def subscribe_breakouts(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for VCP breakout events."""
        self._breakout_cbs.append(callback)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        all_tokens: list[int],
        watchlist_tokens: list[int] | None = None,
    ) -> None:
        """
        Start the WebSocket and tick processor.  Non-blocking.

        all_tokens       : all 500 instrument tokens (MODE_LTP)
        watchlist_tokens : VCP candidates (MODE_QUOTE — includes live OHLC)
        """
        watchlist_tokens = watchlist_tokens or []

        self._processor = _TickProcessor(
            buffer=self._buffer,
            token_to_symbol=self._token_map,
            bar_agg=self._bar_agg,
            vcp_checker=self._vcp_checker,
        )
        self._processor.start()

        ticker = KiteTicker(self._api_key, self._access_token)
        self._ticker = ticker

        def on_ticks(ws: KiteTicker, ticks: list[dict]) -> None:
            # Hot path — must return immediately.
            # deque.append is thread-safe and O(1).
            self._buffer.extend(ticks)

        def on_connect(ws: KiteTicker, response: dict) -> None:
            log.info("live_feed_connected", total_tokens=len(all_tokens))

            # Subscribe all 500 at minimum fidelity (LTP only, ~7 bytes/tick)
            ws.subscribe(all_tokens)
            ws.set_mode(ws.MODE_LTP, all_tokens)

            # Upgrade watchlist to QUOTE (live OHLC, ~57 bytes/tick)
            if watchlist_tokens:
                ws.set_mode(ws.MODE_QUOTE, watchlist_tokens)
                log.info("watchlist_upgraded_to_quote", count=len(watchlist_tokens))

        def on_error(ws: KiteTicker, code: int, reason: str) -> None:
            log.error("live_feed_error", code=code, reason=reason)

        def on_close(ws: KiteTicker, code: int, reason: str) -> None:
            log.warning("live_feed_closed", code=code, reason=reason)

        def on_reconnect(ws: KiteTicker, attempts: int) -> None:
            log.warning("live_feed_reconnecting", attempt=attempts)
            if attempts > 10:
                log.error("live_feed_max_retries_exceeded")
                ws.stop()

        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_error = on_error
        ticker.on_close = on_close
        ticker.on_reconnect = on_reconnect

        # threaded=True → WebSocket runs in its own daemon thread
        threading.Thread(
            target=lambda: ticker.connect(threaded=False),
            daemon=True,
            name="kite-websocket",
        ).start()

        log.info("live_feed_started", tokens=len(all_tokens))

    def stop(self) -> None:
        if self._processor:
            self._processor.stop()
        if self._ticker:
            self._ticker.stop()
        log.info("live_feed_stopped")

    # ------------------------------------------------------------------
    # Bar-close handler
    # ------------------------------------------------------------------

    def _on_bar_close(self, symbol: str, bar: _Bar) -> None:
        """Called by BarAggregator when a 5-min bar closes."""
        bar_dict = bar.to_dict(symbol)

        # 1. Publish to Redis pub/sub so any consumer can react
        try:
            self._r.publish(_PUBSUB_BARS_5MIN, json.dumps(bar_dict))
        except Exception as exc:
            log.error("pubsub_publish_failed", channel=_PUBSUB_BARS_5MIN, error=str(exc))

        # 2. Persist bar to TimescaleDB asynchronously
        threading.Thread(
            target=self._write_bar_to_db,
            args=(symbol, bar),
            daemon=True,
        ).start()

        # 3. Check for VCP breakout on this completed bar
        alert = self._vcp_checker.check(symbol, bar)
        if alert:
            self._fire_breakout(alert)

    def _fire_breakout(self, alert: dict) -> None:
        log.info(
            "vcp_breakout_detected",
            symbol=alert["symbol"],
            price=alert["price"],
            pivot=alert["pivot"],
            volume_ratio=alert["volume_ratio"],
        )
        try:
            self._r.publish(_PUBSUB_BREAKOUT_VCP, json.dumps(alert))
        except Exception as exc:
            log.error("breakout_publish_failed", error=str(exc))

        for cb in self._breakout_cbs:
            try:
                cb(alert)
            except Exception as exc:
                log.error("breakout_callback_error", error=str(exc))

    def _write_bar_to_db(self, symbol: str, bar: _Bar) -> None:
        token = next((t for t, s in self._token_map.items() if s == symbol), None)
        if token is None:
            return
        row = pd.DataFrame(
            [
                {
                    "time": bar.ts,
                    "token": token,
                    "symbol": symbol,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "interval": "5minute",
                }
            ]
        )
        try:
            write_ohlcv(row)
        except Exception as exc:
            log.error("bar_db_write_failed", symbol=symbol, error=str(exc))


# ---------------------------------------------------------------------------
# Redis pub/sub consumer helper
# ---------------------------------------------------------------------------


def subscribe_channel(
    channel: str,
    callback: Callable[[dict], None],
    run_in_background: bool = True,
) -> threading.Thread | None:
    """
    Subscribe to a Redis pub/sub channel and call *callback* for each message.

    Parameters
    ----------
    channel            : one of bars:5min / bars:day / breakout:vcp
    callback           : receives the decoded JSON dict
    run_in_background  : if True, runs in a daemon thread and returns it

    Example
    -------
        def on_breakout(msg):
            print(f"BREAKOUT {msg['symbol']} @ {msg['price']}")

        subscribe_channel("breakout:vcp", on_breakout)
    """

    def _listen() -> None:
        r = get_redis()
        ps = r.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(channel)
        log.info("pubsub_subscribed", channel=channel)
        for message in ps.listen():
            if message["type"] != "message":
                continue
            try:
                callback(json.loads(message["data"]))
            except Exception as exc:
                log.error("pubsub_callback_error", channel=channel, error=str(exc))

    if run_in_background:
        t = threading.Thread(target=_listen, daemon=True, name=f"sub-{channel}")
        t.start()
        return t
    _listen()
    return None
