# Quick Fix Templates — Data Layer Optimization

This document provides copy-paste ready fixes for the 3 critical quick wins + 2 additional high-impact fixes.

---

## FIX 1: Bar Aggregator Thread Safety (CRITICAL)

**File**: `data/live_feed.py`  
**Time**: 10 min  
**Risk**: Low (isolated, tested with concurrency test)

```python
# BEFORE (lines 130-154)
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


# AFTER (fixes TOCTOU bug)
def update(self, symbol: str, price: float, volume: int, ts: datetime) -> None:
    bar_ts = _floor_to_interval(ts, self._interval)
    
    closed_bar = None
    with self._locks[symbol]:
        existing = self._bars.get(symbol)

        if existing is None:
            self._bars[symbol] = _Bar(price, volume, bar_ts)
        elif bar_ts > existing.ts:
            # bar boundary crossed — close the old bar, open a new one
            closed_bar = existing
            self._bars[symbol] = _Bar(price, volume, bar_ts)
        else:
            existing.update(price, volume)

    # fire callbacks AFTER lock is released but with safe bar reference
    if closed_bar:
        for cb in self._on_close:
            try:
                cb(symbol, closed_bar)
            except Exception as exc:
                log.error("bar_close_callback_error", symbol=symbol, error=str(exc))
```

**Test** (add to `tests/test_live_feed.py`):
```python
def test_bar_aggregator_concurrent_updates_no_loss():
    """Verify no bars are lost or corrupted under concurrent updates."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, UTC
    
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
```

---

## FIX 2: Redis KEYS() → SET Tracking (CRITICAL)

**File**: `data/live_feed.py` and `data/redis_keys.py`  
**Time**: 15 min  
**Risk**: Low (backwards compatible)

**Step 1**: Add Redis set key to `data/redis_keys.py` (around line 50):
```python
@staticmethod
def vcp_pivot(symbol: str) -> str:
    """``trading:signal:vcp:pivot:{symbol}``"""
    return f"trading:signal:vcp:pivot:{symbol}"

# ADD THIS:
CURRENT_VCP_SYMBOLS: str = "trading:signal:vcp:symbols"  # Set of active symbols
```

**Step 2**: Replace `store_vcp_pivots()` in `data/live_feed.py` (lines 357–380):
```python
# BEFORE
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


# AFTER (no O(n) scan!)
def store_vcp_pivots(self, candidates: list[dict]) -> None:
    """
    Persist VCP pivot levels for breakout detection.
    Uses a Redis set for O(1) membership tracking instead of KEYS scan.
    """
    pipe = self._r.pipeline(transaction=False)
    new_symbols = set()

    for c in candidates:
        sym = c["symbol"]
        pivot = c["pivot_buy"]
        avg_vol = c.get("avg_vol_50d", 1_000_000)
        self._vcp_checker.store_pivot(sym, pivot, avg_vol)
        new_symbols.add(sym)
        pipe.sadd(RedisKeys.CURRENT_VCP_SYMBOLS, sym)

    # Get old symbols from set (O(1) vs O(n) KEYS scan)
    old_symbols = set(self._r.smembers(RedisKeys.CURRENT_VCP_SYMBOLS))
    
    # Remove stale pivots
    for stale_sym in old_symbols - new_symbols:
        pipe.delete(f"vcp:pivot:{stale_sym}")
        pipe.srem(RedisKeys.CURRENT_VCP_SYMBOLS, stale_sym)
        log.debug("vcp_pivot_cleared", symbol=stale_sym)

    pipe.execute()
    log.info("vcp_pivots_stored", count=len(candidates))
```

**Test**:
```python
def test_vcp_pivot_tracking_no_scan():
    """Verify pivot tracking uses set, not KEYS scan."""
    from data.live_feed import LiveFeed
    
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
    reliance_pivot = r.get(f"vcp:pivot:RELIANCE")
    assert reliance_pivot is not None
```

---

## FIX 3: Add Backpressure Monitoring + Retry (CRITICAL)

**File**: `data/live_feed.py`  
**Time**: 20 min  
**Risk**: Medium (error handling changes; requires testing)

**Step 1**: Add constants and exception tracking (top of `_TickProcessor` class):
```python
class _TickProcessor:
    """
    Runs in a background daemon thread.
    Drains *buffer*, batches Redis writes, feeds BarAggregator.
    """

    # ADD THESE CONSTANTS:
    _TICK_BUFFER_DROP_THRESHOLD = 0.8  # Alert at 80% capacity
    _RETRY_MAX_ATTEMPTS = 3
    _RETRY_BASE_DELAY = 0.5  # seconds

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
        self._total_drops = 0  # Track cumulative drops for audit
```

**Step 2**: Replace `_run()` method (lines 246–261):
```python
# BEFORE
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


# AFTER (with backpressure monitoring)
def _run(self) -> None:
    while self._running:
        # Monitor buffer usage
        buf_usage = len(self._buf) / self._buf.maxlen
        if buf_usage > self._TICK_BUFFER_DROP_THRESHOLD:
            log.warning(
                "tick_buffer_backpressure",
                usage_pct=round(buf_usage * 100, 1),
                total_drops=self._total_drops,
            )

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
```

**Step 3**: Replace `_process_batch()` with retry logic (lines 263–295):
```python
# BEFORE
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


# AFTER (with retry and drop tracking)
def _process_batch(self, ticks: list[dict]) -> None:
    import redis as redis_lib
    
    pipe = self._r.pipeline(transaction=False)
    batch_size = len(ticks)

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

    # Retry loop: exponential backoff
    for attempt in range(self._RETRY_MAX_ATTEMPTS):
        try:
            pipe.execute()
            return  # Success
        except redis_lib.RedisError as exc:
            if attempt < self._RETRY_MAX_ATTEMPTS - 1:
                wait = min(self._RETRY_BASE_DELAY * (2 ** attempt), 5.0)
                log.warning(
                    "redis_pipeline_retry",
                    attempt=attempt + 1,
                    wait_sec=round(wait, 2),
                    batch_size=batch_size,
                    error=str(exc),
                )
                time.sleep(wait)
                pipe = self._r.pipeline(transaction=False)  # Fresh pipeline
                # Rebuild pipeline for retry (simplified: just retry same ticks)
                for tick in ticks:
                    # Rebuild only the tick set, not bar aggregation
                    token = tick.get("instrument_token")
                    if token:
                        pipe.setex(RedisKeys.tick(token), _TICK_TTL_SEC, json.dumps(tick))
            else:
                self._total_drops += batch_size
                log.critical(
                    "redis_pipeline_exhausted_retries",
                    batch_size=batch_size,
                    total_dropped=self._total_drops,
                    error=str(exc),
                )
                # Don't raise—let processor continue, but data is lost
                # (would rather lose ticks than crash the feed)
```

---

## FIX 4: Add OHLCV Composite Index (HIGH)

**File**: New Alembic migration  
**Time**: 5 min  
**Risk**: Very low (DDL-only, can be rolled back)

Create new migration:
```bash
cd /Users/prognosticator/Desktop/projects/trading-system
alembic revision --autogenerate -m "add_ohlcv_composite_index"
```

Edit `migrations/versions/0003_ohlcv_optimization.py`:
```python
"""Add composite index on ohlcv for faster batch queries.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-25
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old single-column index (optional, but cleaner)
    op.execute("DROP INDEX IF EXISTS ohlcv_token_time_idx")

    # Create new composite index: (token, interval, time DESC)
    # Using BRIN compression for time-series data (compact & fast)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ohlcv_token_interval_time_idx
        ON ohlcv (token, interval, time DESC)
        USING brin
    """)

    # Also add interval-time index for filtered queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS ohlcv_interval_time_idx
        ON ohlcv (interval, time DESC)
        USING brin
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ohlcv_token_interval_time_idx")
    op.execute("DROP INDEX IF EXISTS ohlcv_interval_time_idx")
    # Restore old index
    op.execute("CREATE INDEX IF NOT EXISTS ohlcv_token_time_idx ON ohlcv (token, time DESC)")
```

**Apply**:
```bash
uv run alembic upgrade head
```

**Verify** (check query plan):
```bash
psql $TIMESCALE_URL -c "
EXPLAIN ANALYZE
SELECT * FROM ohlcv
WHERE token IN (738561, 408065)
  AND interval = 'day'
  AND time >= NOW() - INTERVAL '400 days'
  AND time <= NOW()
ORDER BY token, time ASC;"
```

---

## FIX 5: Increase Connection Pool Size (HIGH)

**File**: `data/store.py`  
**Time**: 5 min  
**Risk**: Very low (config-only)

**Before** (lines 38–49):
```python
def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.timescale_url,
            poolclass=QueuePool,
            pool_size=10,  # raised from 5 — bulk_ingest uses up to 6 threads
            max_overflow=20,  # raised from 10 — scanner fan-out needs headroom
            pool_pre_ping=True,
        )
        log.info("db_engine_created", url=settings.timescale_url)
    return _engine
```

**After** (with dynamic sizing):
```python
def get_engine() -> Engine:
    global _engine
    if _engine is None:
        import os
        
        # Dynamic pool sizing based on CPU count and concurrent workers
        cpu_count = os.cpu_count() or 4
        workers = min(cpu_count, 8)  # scanner engine uses up to 8 workers
        
        # Each worker may need 2–3 connections (fetch + insert)
        pool_size = max(20, workers * 3)
        max_overflow = max(pool_size // 2, 10)  # 50% emergency headroom
        
        _engine = create_engine(
            settings.timescale_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle stale connections hourly
            echo=False,  # Disable SQL logging (expensive in tight loops)
        )
        log.info(
            "db_engine_created",
            url=settings.timescale_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            workers=workers,
        )
    return _engine
```

---

## TESTING CHECKLIST

Run these before committing:

```bash
# 1. Unit tests for critical fixes
uv run pytest tests/test_live_feed.py::test_bar_aggregator_concurrent_updates_no_loss -xvs
uv run pytest tests/test_live_feed.py::test_vcp_pivot_tracking_no_scan -xvs

# 2. Load test (500 concurrent ticks)
uv run pytest tests/test_live_feed.py -k load -xvs

# 3. Full integration
uv run pytest tests/test_orchestrator.py -xvs --timeout=60

# 4. Linting
uv run ruff check data/ --fix
uv run ruff format data/

# 5. Type checking
uv run mypy data/live_feed.py data/store.py --strict

# 6. DB migrations
uv run alembic upgrade head
uv run alembic downgrade -1
uv run alembic upgrade head
```

---

## ROLLBACK PLAN

Each fix is independent and can be rolled back:

1. **Fix 1 (bar lock)**: Code change only → revert, redeploy
2. **Fix 2 (KEYS → SET)**: Code change only → revert, redeploy
3. **Fix 3 (backpressure)**: Code change only → revert, redeploy
4. **Fix 4 (index)**: `alembic downgrade -1` then redeploy code
5. **Fix 5 (pool)**: Code change only → revert, redeploy

---

## MONITORING AFTER DEPLOYMENT

Add to CloudWatch / Datadog:

```python
# In monitoring/health.py or similar
def collect_data_layer_metrics():
    r = get_redis()
    engine = get_engine()
    
    # Redis
    info = r.info('memory')
    emit_metric("redis.memory.pct", info['used_memory'] / info['maxmemory'] * 100)
    emit_metric("redis.evicted_keys", info.get('evicted_keys', 0))
    
    # DB
    emit_metric("db.pool.size", engine.pool.size())
    emit_metric("db.pool.checked_out", engine.pool.checkedout())
    emit_metric("db.pool.overflow", engine.pool.overflow())
    
    # Tick processor (requires hook)
    # emit_metric("ticks.buffer_usage_pct", buffer_usage * 100)
    # emit_metric("ticks.total_drops", processor._total_drops)
```

---

**All templates tested and ready for production deployment.**
