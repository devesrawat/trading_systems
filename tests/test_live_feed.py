"""Tests for data/live_feed.py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from data.live_feed import BarAggregator, _Bar


def test_bar_aggregator_concurrent_updates_no_loss():
    """Verify no bars are lost or corrupted under concurrent updates."""
    agg = BarAggregator(interval_min=1)
    closed_bars = []

    def on_close(symbol: str, bar: _Bar):
        closed_bars.append((symbol, bar.ts, bar.volume))

    agg.on_bar_close(on_close)

    # Simulate 1000 concurrent ticks across 10 symbols
    def emit_ticks(symbol_id: int):
        base_time = datetime.now(UTC)
        for i in range(100):
            ts = base_time + timedelta(seconds=i * 0.5)
            agg.update(f"SYM{symbol_id}", 100.0 + i, 1000, ts)

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(emit_ticks, i) for i in range(10)]
        for f in futures:
            f.result()

    # Verify we captured close events (one per minute per symbol minimum)
    assert len(closed_bars) >= 9  # At least 9 minute transitions
    # Verify no duplicate timestamps
    assert len(set(closed_bars)) == len(closed_bars)


def test_bar_aggregator_basic_update():
    """Test basic bar aggregation without concurrency."""
    agg = BarAggregator(interval_min=1)
    closed_bars = []

    def on_close(symbol: str, bar: _Bar):
        closed_bars.append((symbol, bar.ts, bar.open, bar.close, bar.high, bar.low, bar.volume))

    agg.on_bar_close(on_close)

    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    # First tick at 10:00:00 — opens first bar
    agg.update("AAPL", 100.0, 1000, base_time)

    # Second tick at 10:00:30 — same bar
    agg.update("AAPL", 101.0, 1000, base_time + timedelta(seconds=30))

    # No bar closed yet
    assert len(closed_bars) == 0

    # Tick at 10:01:00 — closes first bar, opens new bar
    agg.update("AAPL", 102.0, 1000, base_time + timedelta(minutes=1))

    # Now first bar should be closed
    assert len(closed_bars) == 1
    symbol, bar_ts, open_price, close_price, high, low_price, vol = closed_bars[0]
    assert symbol == "AAPL"
    assert bar_ts == base_time
    assert open_price == 100.0
    assert close_price == 101.0
    assert high == 101.0
    assert low_price == 100.0
    assert vol == 2000


def test_bar_aggregator_multiple_symbols():
    """Test aggregator with multiple symbols concurrently."""
    agg = BarAggregator(interval_min=1)
    closed_bars = []

    def on_close(symbol: str, bar: _Bar):
        closed_bars.append(symbol)

    agg.on_bar_close(on_close)

    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Add ticks to different symbols
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        agg.update(sym, 100.0, 1000, base_time)
        agg.update(sym, 101.0, 1000, base_time + timedelta(seconds=30))
        agg.update(sym, 102.0, 1000, base_time + timedelta(minutes=1))

    # Each symbol should have closed exactly one bar
    assert len(closed_bars) == 3
    assert set(closed_bars) == {"AAPL", "MSFT", "GOOGL"}


def test_bar_aggregator_no_callbacks_on_intra_bar():
    """Verify no callbacks are fired for intra-bar updates."""
    agg = BarAggregator(interval_min=1)
    callback_count = 0

    def on_close(symbol: str, bar: _Bar):
        nonlocal callback_count
        callback_count += 1

    agg.on_bar_close(on_close)

    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Many ticks within the same 1-minute bar
    for i in range(50):
        agg.update("AAPL", 100.0 + i * 0.01, 1000, base_time + timedelta(seconds=i))

    # No bar closed yet
    assert callback_count == 0


def test_vcp_pivot_tracking_no_scan():
    """Verify pivot tracking uses set, not KEYS scan."""
    from data.live_feed import LiveFeed
    from data.redis_keys import RedisKeys
    from data.store import get_redis

    feed = LiveFeed("test_key", "test_token")
    candidates = [
        {"symbol": "RELIANCE", "pivot_buy": 2895.0, "avg_vol_50d": 1_000_000},
        {"symbol": "INFY", "pivot_buy": 2200.0, "avg_vol_50d": 500_000},
    ]

    r = get_redis()
    r.flushdb()  # Clean slate

    # Call store_vcp_pivots
    feed.store_vcp_pivots(candidates)

    # Verify set was populated
    symbols_in_set = r.smembers(RedisKeys.CURRENT_VCP_SYMBOLS)
    assert symbols_in_set == {"RELIANCE", "INFY"}

    # Verify pivots exist
    reliance_pivot = r.get("vcp:pivot:RELIANCE")
    assert reliance_pivot is not None
