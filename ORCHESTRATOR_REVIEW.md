# Deep Technical Review: Orchestration & Scheduling Layer

**Status**: Trading system running 24/7. Critical findings that pose capital and operational risks.

**Reviewed Components**:
- `orchestrator/main.py` — TradingSystem (1000+ LOC)
- `orchestrator/scheduler.py` — TradingScheduler + APScheduler jobs (607 LOC)
- `orchestrator/runner.py` — OrchestratorRunner (phase 7 integration)
- `orchestrator/ab_tester.py` — A/B test orchestration
- `monitoring/reporters.py` — Daily/weekly/monthly reporting
- `audit/persistence.py` — Audit log storage

---

## Critical Issues

### ISSUE 1: APScheduler Not Configured for High-Availability
**Location**: `orchestrator/scheduler.py:52`

**Severity**: **CRITICAL**

**Current behavior**:
```python
self._scheduler = BackgroundScheduler(timezone=_TZ_IST)
# ... no thread pool config, no misfire policy, no job store
```
- Default `BackgroundScheduler` uses 1 worker thread (no concurrency)
- No `jobstore` configured → jobs lost on process crash
- No `misfire_grace_time` → missed jobs discarded silently
- No `coalesce` or `max_instances` protection → duplicate job execution possible
- Jobs are held only in memory; single scheduler process = single point of failure

**Impact**:
- **Daily trading loops skipped silently** if scheduler delays 1+ minutes
- **Pre-market setups missed** (universe not loaded, model not loaded, sentiment not cached)
- **Post-market summaries lost** (no daily audit trail)
- **Weekly resets not executed** (circuit breaker state carries over)
- **Retrain checks skipped** (model drift undetected for weeks)
- Restart/crash = all in-flight jobs vanish with no persistence

**Proposed fix**:
```python
from apscheduler.jobstores.memory import MemoryJobStore  # temp; upgrade to TimescaleDB
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.thread import ThreadPoolExecutor
from apscheduler.executors.pool import ProcessPoolExecutor

# High-availability config
self._scheduler = BackgroundScheduler(
    jobstores={
        "default": MemoryJobStore(),  # Phase 7.5: migrate to SQLAlchemy job store (TimescaleDB)
    },
    executors={
        "default": ThreadPoolExecutor(max_workers=4),  # Parallel job execution
        "processpool": ProcessPoolExecutor(max_workers=2),  # For expensive compute jobs
    },
    job_defaults={
        "coalesce": True,  # If N executions missed, only run once
        "max_instances": 1,  # Prevent duplicate concurrent execution
        "misfire_grace_time": 30,  # Execute missed jobs if < 30s late
    },
    timezone=_TZ_IST,
)
```

**Complexity**: 3/5 (moderate refactor, requires testing concurrent job behavior)

---

### ISSUE 2: Silent Job Failures — No Alerting on Missed Windows
**Location**: `orchestrator/scheduler.py:274-293` (wrapper `_safe` method)

**Severity**: **CRITICAL**

**Current behavior**:
```python
def _safe(self, fn):
    """Wrap a job function so exceptions alert Telegram but never crash."""
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            log.error("scheduler_job_failed", job=fn.__name__, error=str(exc))
            try:
                TelegramAlerter().alert_system_error(...)
            except Exception:
                pass  # Silent fail if Telegram is down
```
- ✅ Catches exceptions
- ❌ No tracking of **missed execution windows** (job scheduled but never ran)
- ❌ No tracking of **job duration** (how long jobs actually take)
- ❌ No alerting if a job was skipped due to `coalesce=True` silently
- ❌ Telegram alerter wrapped in bare `except` → errors swallowed

**Impact**:
- Trading loop scheduled for every 5 min but runs in 2 min → no visible issue
- But if ONE execution takes >5 min (e.g., broker API slow), the next scheduled slot is **coalesced/skipped** with no alert
- System could be missing 25 min of signals (5 × 5-min slots) with no indication

**Proposed fix**:
```python
import time
from monitoring.health import HealthMonitor

def _safe(self, fn):
    """Wrap job with execution tracking and deadline enforcement."""
    def wrapper(*args, **kwargs):
        job_name = getattr(fn, "__name__", "unknown")
        start_time = time.time()
        duration_threshold_sec = 300  # e.g., 5-min trading loop shouldn't exceed 4 min
        
        try:
            fn(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Log job execution time
            log.info(
                "scheduler_job_executed",
                job=job_name,
                duration_sec=elapsed,
            )
            
            # Alert if job exceeded safe duration
            if elapsed > duration_threshold_sec:
                log.warning(
                    "scheduler_job_slow",
                    job=job_name,
                    duration_sec=elapsed,
                    threshold_sec=duration_threshold_sec,
                )
                HealthMonitor().record_slow_job(job_name, elapsed)
                
        except Exception as exc:
            elapsed = time.time() - start_time
            log.error(
                "scheduler_job_failed",
                job=job_name,
                duration_sec=elapsed,
                error=str(exc),
            )
            
            # Alert with retry guidance
            try:
                TelegramAlerter().alert_system_error(
                    module=job_name,
                    error_msg=f"{str(exc)} (after {elapsed:.1f}s)",
                )
            except Exception as tg_err:
                log.error("telegram_alerter_failed", error=str(tg_err))
                # Don't swallow the error; record it for debugging
    
    wrapper.__name__ = getattr(fn, "__name__", "unknown")
    return wrapper
```

**Complexity**: 2/5 (add tracking + HealthMonitor logging)

---

### ISSUE 3: Overlapping Equity + Crypto Cycles (Race Condition Risk)
**Location**: `orchestrator/main.py:367-384` (trading_loop) and `scheduler.py:56-59` (registration)

**Severity**: **HIGH**

**Current behavior**:
```python
# Main trading loop — called every 5 min
def trading_loop(self) -> None:
    # Equity cycle (runs only if market is equity or both)
    if self._market_type in ("equity", "both"):
        self._update_regime()
        if not should_suppress_new_entries():
            self._run_equity_cycle()  # Can hold broker lock
    
    # Crypto cycle (always runs)
    if self._market_type in ("crypto", "both"):
        self._run_crypto_cycle()  # Also tries to hold broker lock
```
- Both cycles use shared `self._executor` (OrderExecutor)
- Both call `self._broker.get_balance()` (single broker connection)
- Equity cycle holds regime lock + potentially broker locks
- Crypto cycle starts immediately after without wait
- **Race condition**: Both cycles try to place orders simultaneously

**Impact**:
- Two orders placed in parallel, broker rejects one due to insufficient capital
- Risk checks pass independently but together exceed daily_dd_limit
- Portfolio state becomes inconsistent (equity thinks it owns 100 shares, crypto thinks it owns 50 coins)

**Proposed fix**:
```python
from threading import Lock

class TradingSystem:
    def __init__(self, ...):
        # ... existing init ...
        self._trading_lock = Lock()  # Serialize cycle execution
        
    def trading_loop(self) -> None:
        # ... existing setup ...
        
        try:
            # Only ONE cycle can execute at a time
            with self._trading_lock:
                if self._market_type in ("equity", "both"):
                    self._update_regime()
                    if not self._regime_detector.should_suppress_new_entries(...):
                        self._run_equity_cycle()
                
                if self._market_type in ("crypto", "both"):
                    self._run_crypto_cycle()
```

**Alternatively (better for scaling)**:
- Launch equity and crypto cycles in separate threads with independent locks
- Use a shared `CapitalReserver` to prevent over-allocation between cycles

**Complexity**: 2/5 (add threading lock)

---

### ISSUE 4: State Accumulation Between Jobs (Memory Leaks)
**Location**: `orchestrator/main.py:213-223` (universe cache + state)

**Severity**: **HIGH**

**Current behavior**:
```python
self._vcp_candidates: list[dict] = []  # Raw VCP results
self._open_positions: set[str] = set()  # Open positions
self._pre_market_signals: list[Signal] = []  # Collected signals (in runner.py:79)

# In pre_market_setup():
self._vcp_candidates.append(result_dict)  # Appends, never clears

# In post_market_summary():
self._open_positions.clear()  # Cleared at end of day
```
- `_vcp_candidates` is **never cleared** between market sessions
- If pre_market runs multiple times (e.g., restart, or in "both" mode) → list grows without bound
- Over 1 month: **thousands of duplicate results accumulate**
- Each signal stored in memory is 100+ bytes × thousands = **memory bloat**

**Impact**:
- Memory usage grows unbounded (crash if running 24/7)
- Duplicate signal processing (same candidate processed multiple times)
- Portfolio state inconsistencies
- Eventual OOM kill

**Proposed fix**:
```python
def pre_market_setup(self) -> None:
    """Prepare the system for the trading session."""
    log.info("pre_market_setup_start", market=self._market_type)
    
    # Clear prior session state
    self._vcp_candidates.clear()
    self._pre_market_signals.clear()  # (if using runner.py)
    
    # ... rest of setup ...
```

**Complexity**: 1/5 (one-line fix)

---

### ISSUE 5: Blocking Audit Logging During Trade Execution
**Location**: `audit/persistence.py:137-174` (all logging methods) + `execution/logger.py` (if called synchronously)

**Severity**: **HIGH**

**Current behavior**:
```python
# In execution path (e.g., after order placement):
self.trade_logger.log_signal(symbol, features, score, action, strategy_version)

# In audit/persistence.py:
@staticmethod
def log_signal(entry: SignalLogEntry) -> None:
    data = entry.model_dump_json()  # Serialization
    key = f"{RedisKeys.AUDIT_SIGNALS}:{entry.signal_id}"
    get_redis().set(key, data, ex=TTL_SECONDS)  # Network I/O to Redis
    log.debug("audit_signal_logged", signal_id=entry.signal_id)
```
- Redis `.set()` is **synchronous and blocking**
- If Redis is slow (5–50ms latency), order execution waits
- During fast market moves, **every 50ms delay = missed signal window**
- Under load, multiple threads wait on same Redis connection

**Impact**:
- Trading loop cycle time increases (designed 5 min × N symbols = should be <60s, but if N=100 + 50ms per log = 5s extra)
- If 10 concurrent orders wait on Redis → 500ms+ blocked
- Execution of next signal delayed → missed opportunities

**Proposed fix**:
```python
from queue import Queue
from threading import Thread

class AuditLogger:
    """Async audit logging with background thread."""
    
    _log_queue: Queue = Queue(maxsize=10000)  # Buffer up to 10k logs
    
    @classmethod
    def start_background_writer(cls):
        """Start the background audit writer thread."""
        Thread(target=cls._background_write, daemon=True).start()
    
    @staticmethod
    def log_signal(entry: SignalLogEntry) -> None:
        """Queue signal for async logging (non-blocking)."""
        try:
            AuditLogger._log_queue.put_nowait(("signal", entry))
        except:
            log.warning("audit_queue_full")  # Silent drop if backed up
    
    @classmethod
    def _background_write(cls):
        """Background thread: write buffered logs to Redis/DB."""
        while True:
            try:
                log_type, entry = cls._log_queue.get(timeout=1)
                
                if log_type == "signal":
                    _AuditPersistence.log_signal(entry)
                # ... handle other types ...
                
            except Empty:
                continue
            except Exception as e:
                log.error("audit_write_failed", error=str(e))
```

**Complexity**: 3/5 (requires queue + background thread + graceful shutdown)

---

### ISSUE 6: Scheduler Job Ordering — Post-Market Depends on Trading Loop Completion
**Location**: `orchestrator/scheduler.py:92-112` (equity jobs) and main.py event flow

**Severity**: **MEDIUM**

**Current behavior**:
```
# Equity schedule (IST weekdays):
09:15        → trading_loop (first execution)
09:20        → trading_loop (2nd execution)
...
15:25        → trading_loop (last execution)
15:35        → post_market_summary  # Scheduled to run 10 min after last trading_loop
```
- `post_market_summary()` is hard-coded to 15:35 IST
- But `trading_loop()` at 15:25 may take 30 seconds or more to complete
- If 15:25 execution finishes at 15:25:45, the scheduler still tries to run `post_market_summary` at 15:35:00 (not after 15:25 job completes)

**Implied constraint**: `post_market_summary()` must run AFTER the final `trading_loop()` completes, not just at a fixed time.

**Impact**:
- `post_market_summary()` runs while orders from 15:25 are still settling
- Daily summary is incomplete (includes in-flight orders)
- Audit trail has gaps

**Proposed fix**:
```python
def trading_loop(self) -> None:
    """One scan-signal-execute cycle."""
    try:
        # ... main loop logic ...
        
    finally:
        # Always run at the end of trading loop
        HealthMonitor().write_heartbeat()
        
        # If this is the last trading loop of the day (determined by current time)
        # we could trigger post_market_summary
        # But better: let the scheduler handle timing, just ensure logging is complete

# Or: make trading_loop idempotent and let post_market_summary be called after

# Scheduler config: Add a dependency mechanism
self._scheduler.add_job(
    func=self._safe(s.post_market_summary),
    trigger=CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone=_TZ_IST),
    id="equity_post_market",
    # Check if the 15:25 job has completed before running
    # (APScheduler doesn't support this natively; need custom check)
)
```

**Better approach**: Use `TradingSystem.trading_loop()` to detect it's the last execution of the day and call `post_market_summary()` directly:

```python
def trading_loop(self) -> None:
    """One scan-signal-execute cycle."""
    now = datetime.now(tz=IST)
    is_last_equity_cycle = now.hour == 15 and now.minute in (20, 25)  # Last scheduled slot
    
    try:
        # ... main trading logic ...
    finally:
        HealthMonitor().write_heartbeat()
        
        if is_last_equity_cycle and self._market_type in ("equity", "both"):
            # Defer post_market_summary to next scheduler tick, or run inline
            self.post_market_summary()
```

**Complexity**: 2/5 (add end-of-day detection + optional call)

---

### ISSUE 7: A/B Test State Race Condition (Random Seed Not Seeded Per-Symbol)
**Location**: `orchestrator/ab_tester.py:67-89` (route_signal_to_model)

**Severity**: **MEDIUM**

**Current behavior**:
```python
def route_signal_to_model(self, symbol: str) -> str:
    """Route a signal to either champion or challenger (50/50 random)."""
    choice = self._random.choice(["champion", "challenger"])
    return choice
```
- Uses `self._random` (a `random.Random()` instance) initialized once
- **No seed per-symbol** → same symbol might always route to champion
- **No per-day seed** → same symbol always gets same model for the day (not truly random)
- In "both" market mode, if equity and crypto cycles run near-simultaneously, they might race on `self._random`

**Impact**:
- A/B test is not truly 50/50 → statistical test will fail
- One model (e.g., champion) gets systematically easier symbols
- Comparison metrics are biased

**Proposed fix**:
```python
import hashlib
from datetime import datetime

def route_signal_to_model(self, symbol: str) -> str:
    """Route a signal to champion or challenger (reproducible 50/50)."""
    # Use symbol + date to seed the decision
    # This ensures: same symbol gets same model throughout the day,
    # but different symbols get different models (random), and next day symbols rotate
    
    today = datetime.utcnow().date()
    seed_str = f"{symbol}:{today}"
    
    # Deterministic hash-based routing (reproducible)
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    choice = "champion" if hash_val % 2 == 0 else "challenger"
    
    log.debug("ab_route_decision", symbol=symbol, model=choice, seed=seed_str)
    return choice
```

**Alternative** (true random with thread safety):
```python
import threading

class ABTestOrchestrator:
    def __init__(self, ...):
        self._random_lock = threading.Lock()
        self._random = random.Random()
    
    def route_signal_to_model(self, symbol: str) -> str:
        with self._random_lock:
            choice = self._random.choice(["champion", "challenger"])
        return choice
```

**Complexity**: 1/5 (simple fix)

---

### ISSUE 8: Synchronous TimescaleDB Writes During Backtesting (Blocking)
**Location**: `orchestrator/ab_tester.py:166-216` (log_to_timescaledb)

**Severity**: **MEDIUM**

**Current behavior**:
```python
def _log_to_timescaledb(self, result: ABTestResult) -> None:
    """Write A/B test result to TimescaleDB for permanent storage."""
    try:
        with self._engine.begin() as conn:  # Synchronous transaction
            conn.execute(text("CREATE TABLE IF NOT EXISTS ab_test_results (...)"))
            conn.execute(text("INSERT INTO ab_test_results (...)"), {...})
    except Exception as e:
        log.error("timescaledb_log_failed", error=str(e))
```
- **CREATE TABLE on every write** (should be done once at startup)
- **Synchronous writes** during trade execution
- If DB is slow or overloaded → order placement blocked

**Impact**:
- Every trade execution waits for a DB round-trip (5–50ms)
- Under load, thread pool gets exhausted waiting on DB

**Proposed fix**:
```python
class ABTestOrchestrator:
    def __init__(self, ...):
        self._engine = get_engine()
        self._redis = get_redis()
        self._ensure_table_exists()  # Call once at init
    
    def _ensure_table_exists(self) -> None:
        """Create table at startup, not on every write."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ab_test_results (
                        id SERIAL PRIMARY KEY,
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        ...
                    );
                    CREATE INDEX IF NOT EXISTS idx_ab_test_time_model 
                        ON ab_test_results(time DESC, model_name);
                """))
        except Exception as e:
            log.warning("table_creation_failed", error=str(e))
    
    def log_ab_test_result(self, ...):
        """Log to Redis immediately (fast), write to DB async."""
        # ... existing Redis logging ...
        
        # Queue for background DB write (don't block)
        self._queue_db_write(result)
    
    def _queue_db_write(self, result: ABTestResult) -> None:
        """Queue result for async DB persistence."""
        import json
        try:
            self._redis.lpush(
                "trading:ab_test:pending_writes",
                json.dumps(result.to_dict()),
            )
        except Exception as e:
            log.warning("queue_db_write_failed", error=str(e))
```

**Complexity**: 2/5 (add background write queue)

---

### ISSUE 9: OrchestratorRunner State Accumulation (runner.py)
**Location**: `orchestrator/runner.py:79` (pre_market_signals)

**Severity**: **MEDIUM**

**Current behavior**:
```python
class OrchestratorRunner:
    def __init__(self, market_type: str | None = None) -> None:
        # ...
        self._pre_market_signals: list[Signal] = []  # Accumulates but never cleared
    
    def pre_market_setup(self) -> None:
        # Signals are appended
        self._pre_market_signals.append(signal)
        
    def trading_loop(self) -> None:
        for signal in self._pre_market_signals:
            self._process_signal(signal)  # Processes all signals
        # But _pre_market_signals is NEVER cleared
```
- Same signals are processed on every `trading_loop()` call
- After 100 trading_loop calls (500 min = ~8 hours), signals are processed 100× each

**Impact**:
- Duplicate trades placed for same signal
- Portfolio overloaded with redundant orders

**Proposed fix**:
```python
def trading_loop(self) -> None:
    """Trading loop: process signals once, then clear."""
    try:
        self.trading_system.trading_loop()
        
        for signal in self._pre_market_signals:
            self._process_signal(signal)
        
        self._pre_market_signals.clear()  # Clear after processing
        
    except Exception as e:
        log.error("trading_loop_error", error=str(e))
```

**Complexity**: 1/5 (one-line fix)

---

### ISSUE 10: Hard-Coded Reporting Schedules May Conflict
**Location**: `orchestrator/scheduler.py:203-237` (reporting jobs)

**Severity**: **MEDIUM**

**Current behavior**:
```python
# Daily reporting — 16:00 IST (post-market)
self._scheduler.add_job(..., id="daily_reporting", trigger=CronTrigger(hour=16, minute=0, day_of_week="mon-fri", ...))

# Weekly reporting — Friday 17:00 IST
self._scheduler.add_job(..., id="weekly_reporting", trigger=CronTrigger(hour=17, minute=0, day_of_week="fri", ...))

# A/B test reporting — Friday 17:05 IST (after weekly report)
self._scheduler.add_job(..., id="ab_test_reporting", trigger=CronTrigger(hour=17, minute=5, day_of_week="fri", ...))

# Monthly reporting — Last day of month at 17:00 IST
self._scheduler.add_job(..., id="monthly_reporting", trigger=CronTrigger(hour=17, minute=0, day="31", ...))
```
- On Friday, 4 jobs try to run nearly simultaneously:
  - 16:00 — daily_reporting
  - 17:00 — weekly_reporting (overlaps with daily if daily runs long)
  - 17:00 — monthly_reporting (if last day of month is Friday)
  - 17:05 — ab_test_reporting
- Each report job likely sends Telegram alerts, potentially flooding the bot
- Report generation is synchronous and can take 10+ seconds

**Impact**:
- Reports may be incomplete (one job cancels another)
- Telegram rate limiting kicks in
- System appears unresponsive

**Proposed fix**:
```python
# Stagger reporting jobs with explicit delays
self._scheduler.add_job(
    func=self._safe(self._daily_reporting),
    trigger=CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone=_TZ_IST),
    id="daily_reporting",
)

self._scheduler.add_job(
    func=self._safe(self._weekly_reporting),
    trigger=CronTrigger(hour=17, minute=0, day_of_week="fri", timezone=_TZ_IST),
    id="weekly_reporting",
)

# Ensure AB test runs 5 min AFTER weekly (not at same time)
self._scheduler.add_job(
    func=self._safe(self._ab_test_reporting),
    trigger=CronTrigger(hour=17, minute=5, day_of_week="fri", timezone=_TZ_IST),
    id="ab_test_reporting",
)

# Monthly reporting: if it lands on Friday, run it at 17:30 IST (after weekly + AB)
# For now, document the conflict
```

**Complexity**: 1/5 (add comments + optional delay adjustment)

---

## High-Priority Issues (Architectural)

### ISSUE 11: No Job Persistence (Jobs Lost on Crash)
**Location**: `orchestrator/scheduler.py:52`

**Severity**: **HIGH** (architectural)

**Current behavior**:
- APScheduler uses `MemoryJobStore` (in-memory only)
- Jobs exist only in RAM; process crash → all jobs vanish
- No restart recovery

**Proposed fix** (Phase 7.5):
```python
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

self._scheduler = BackgroundScheduler(
    jobstores={
        "default": SQLAlchemyJobStore(
            url="postgresql://user:pass@localhost/trading_db",
            # Table: apscheduler_jobs
        )
    },
    # ... rest of config ...
)
```

**Complexity**: 4/5 (requires DB schema + migration)

---

### ISSUE 12: No Circuit Breaker for Reporting Jobs
**Location**: `orchestrator/scheduler.py:203-237` (reporting methods)

**Severity**: **MEDIUM**

**Current behavior**:
- Reporting jobs are CPU-intensive (read entire trade history, compute stats)
- No timeout or circuit breaker
- If report generation hangs, scheduler blocks all subsequent jobs

**Proposed fix**:
```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class TradingScheduler:
    def __init__(self, ...):
        self._report_executor = ThreadPoolExecutor(max_workers=1)  # Dedicated thread for reports
    
    def _safe(self, fn):
        """Wrap job with timeout enforcement."""
        def wrapper(*args, **kwargs):
            job_name = getattr(fn, "__name__", "unknown")
            timeout_sec = 300 if "reporting" in job_name else 60  # 5 min for reports, 1 min for others
            
            try:
                future = self._report_executor.submit(fn, *args, **kwargs)
                future.result(timeout=timeout_sec)
            except TimeoutError:
                log.error(f"scheduler_job_timeout", job=job_name, timeout_sec=timeout_sec)
                TelegramAlerter().alert_system_error(job_name, "Timeout")
            except Exception as e:
                # ... existing error handling ...
        
        return wrapper
```

**Complexity**: 3/5 (add executor + timeout)

---

### ISSUE 13: No Graceful Shutdown (Jobs Interrupted)
**Location**: `orchestrator/scheduler.py:71-73`

**Severity**: **MEDIUM**

**Current behavior**:
```python
def stop(self) -> None:
    self._scheduler.shutdown(wait=True)
    log.info("scheduler_stopped")
```
- `wait=True` waits for all jobs to complete
- But if a job is hung (e.g., waiting for broker API), this blocks indefinitely
- Signals (SIGTERM) can't interrupt a hung job

**Proposed fix**:
```python
import signal
import threading

class TradingScheduler:
    def stop(self, timeout_sec: int = 60) -> None:
        """Gracefully shutdown the scheduler."""
        log.info("scheduler_shutdown_initiated", timeout_sec=timeout_sec)
        
        try:
            # Set a timeout for the shutdown
            def do_shutdown():
                self._scheduler.shutdown(wait=True)
            
            thread = threading.Thread(target=do_shutdown, daemon=True)
            thread.start()
            thread.join(timeout=timeout_sec)
            
            if thread.is_alive():
                log.error("scheduler_shutdown_timeout")
                # Force shutdown (some jobs may be interrupted)
                self._scheduler.shutdown(wait=False)
        except Exception as e:
            log.error("scheduler_shutdown_failed", error=str(e))
```

**Complexity**: 2/5 (add timeout logic)

---

### ISSUE 14: Regime Detection Not Synchronized Between Cycles
**Location**: `orchestrator/main.py:366-380` (regime gating in trading_loop)

**Severity**: **LOW**

**Current behavior**:
```python
def trading_loop(self) -> None:
    if self._market_type in ("equity", "both"):
        self._update_regime()
        if self._regime_detector.should_suppress_new_entries(self._current_regime.state):
            log.info("equity_cycle_skipped_choppy_regime", ...)
        else:
            self._run_equity_cycle()
    
    if self._market_type in ("crypto", "both"):
        self._run_crypto_cycle()  # Always runs, no regime check
```
- Equity cycle can be suppressed due to choppy market regime
- Crypto cycle always runs
- If both are suppressed simultaneously, portfolio might accumulate unhedged crypto positions

**Impact**: Low (crypto doesn't hedge equity, but may increase volatility)

**Proposed fix**: Apply regime gating to crypto as well (if market is truly choppy)

**Complexity**: 1/5 (add regime check to crypto)

---

## Top 3 Quick Wins

### Quick Win 1: Add State Clearing to pre_market_setup (1 hour)
**Fixes**: Issue #4 (state accumulation)

```python
# In orchestrator/main.py pre_market_setup():
self._vcp_candidates.clear()

# In orchestrator/runner.py pre_market_setup():
self._pre_market_signals.clear()

# In orchestrator/runner.py trading_loop():
self._pre_market_signals.clear()  # After processing
```

**Impact**: Prevents memory bloat, ensures clean session state

**Verification**:
```bash
# Monitor memory usage
watch -n 5 'ps aux | grep -E "python|trading" | awk "{print \$6}" | sort -n'
```

---

### Quick Win 2: Fix A/B Test Routing (30 min)
**Fixes**: Issue #7 (race condition in A/B routing)

```python
# In orchestrator/ab_tester.py:
def route_signal_to_model(self, symbol: str) -> str:
    """Route 50/50 based on deterministic hash of symbol + date."""
    import hashlib
    from datetime import datetime
    
    today = datetime.utcnow().date()
    seed_str = f"{symbol}:{today}"
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    choice = "champion" if hash_val % 2 == 0 else "challenger"
    
    log.debug("ab_route_decision", symbol=symbol, model=choice)
    return choice
```

**Impact**: A/B test results are now statistically valid

---

### Quick Win 3: Enhance _safe Wrapper with Duration Tracking (45 min)
**Fixes**: Issue #2 (silent job failures)

```python
# In orchestrator/scheduler.py _safe():
def _safe(self, fn):
    import time
    
    def wrapper(*args, **kwargs):
        job_name = getattr(fn, "__name__", "unknown")
        start_time = time.time()
        
        try:
            fn(*args, **kwargs)
            elapsed = time.time() - start_time
            log.info("scheduler_job_executed", job=job_name, duration_sec=elapsed)
        except Exception as exc:
            elapsed = time.time() - start_time
            log.error(
                "scheduler_job_failed",
                job=job_name,
                duration_sec=elapsed,
                error=str(exc),
            )
            try:
                from monitoring.alerts import TelegramAlerter
                TelegramAlerter().alert_system_error(
                    module=job_name,
                    error_msg=f"{str(exc)} (after {elapsed:.1f}s)",
                )
            except Exception as e:
                log.error("telegram_alerter_failed", error=str(e))
    
    wrapper.__name__ = getattr(fn, "__name__", "unknown")
    return wrapper
```

**Impact**: Better visibility into job execution times; alerts on slow/failing jobs

---

## Top 3 Architectural Changes

### Architectural Change 1: APScheduler High-Availability (3-4 days)
**Fixes**: Issue #1, #11, #13

**Scope**:
1. Upgrade `BackgroundScheduler` config with thread pool + job store
2. Migrate from MemoryJobStore to SQLAlchemyJobStore (TimescaleDB)
3. Add graceful shutdown with timeout
4. Add job execution tracking + metrics

**Code**:
```python
# orchestrator/scheduler.py
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.thread import ThreadPoolExecutor
from apscheduler.executors.pool import ProcessPoolExecutor

class TradingScheduler:
    def __init__(self, system, market_type="equity"):
        self._system = system
        self._market_type = market_type.lower()
        
        # High-availability config
        self._scheduler = BackgroundScheduler(
            jobstores={
                "default": SQLAlchemyJobStore(
                    url=settings.scheduler_db_url,  # e.g., postgresql://...
                ),
            },
            executors={
                "default": ThreadPoolExecutor(max_workers=4),
                "blocking": ProcessPoolExecutor(max_workers=2),
            },
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 30,
            },
            timezone=_TZ_IST,
        )
    
    def stop(self, timeout_sec: int = 60):
        """Graceful shutdown with timeout."""
        import threading
        import signal
        
        log.info("scheduler_shutdown_initiated", timeout_sec=timeout_sec)
        
        def do_shutdown():
            self._scheduler.shutdown(wait=True)
        
        shutdown_thread = threading.Thread(target=do_shutdown, daemon=True)
        shutdown_thread.start()
        shutdown_thread.join(timeout=timeout_sec)
        
        if shutdown_thread.is_alive():
            log.error("scheduler_shutdown_timeout")
            self._scheduler.shutdown(wait=False)
```

**Testing**:
- Verify jobs persist after restart
- Simulate job hang and verify timeout works
- Monitor scheduler uptime over 1 week

---

### Architectural Change 2: Async Audit Logging (2-3 days)
**Fixes**: Issue #5, #8

**Scope**:
1. Create background worker thread for audit writes
2. Replace synchronous Redis/DB calls with async queue
3. Add metrics (queue depth, write latency)
4. Graceful shutdown of background writer

**Code**:
```python
# audit/async_writer.py (new file)
from queue import Queue, Empty
from threading import Thread
import time

class AsyncAuditWriter:
    """Background worker for async audit logging."""
    
    def __init__(self, max_queue_size: int = 10000):
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._worker_thread: Thread | None = None
        self._running = False
        self._metrics = {
            "total_logs": 0,
            "failed_logs": 0,
            "queue_depth": 0,
        }
    
    def start(self):
        """Start the background writer thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = Thread(
            target=self._background_write,
            daemon=True,
            name="AuditWriter",
        )
        self._worker_thread.start()
        log.info("async_audit_writer_started")
    
    def stop(self, timeout_sec: int = 10):
        """Stop the background writer and flush remaining logs."""
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout_sec)
            if self._worker_thread.is_alive():
                log.warning("audit_writer_thread_still_alive")
    
    def queue_log(self, entry_type: str, entry):
        """Queue a log entry for async write (non-blocking)."""
        try:
            self._queue.put_nowait((entry_type, entry))
        except:
            log.warning("audit_queue_full")  # Drop if backed up
            self._metrics["failed_logs"] += 1
    
    def _background_write(self):
        """Background thread: write buffered logs to persistence."""
        from audit.persistence import _AuditPersistence
        
        while self._running or not self._queue.empty():
            try:
                entry_type, entry = self._queue.get(timeout=1)
                
                try:
                    if entry_type == "signal":
                        _AuditPersistence.log_signal(entry)
                    elif entry_type == "trade":
                        _AuditPersistence.log_trade(entry)
                    elif entry_type == "risk_decision":
                        _AuditPersistence.log_risk_decision(entry)
                    elif entry_type == "order":
                        _AuditPersistence.log_order(entry)
                    elif entry_type == "circuit_breaker":
                        _AuditPersistence.log_circuit_breaker(entry)
                    
                    self._metrics["total_logs"] += 1
                    
                except Exception as e:
                    log.error("audit_persistence_failed", error=str(e))
                    self._metrics["failed_logs"] += 1
            
            except Empty:
                continue
            except Exception as e:
                log.error("audit_writer_thread_error", error=str(e))
        
        log.info("audit_writer_thread_exited", metrics=self._metrics)

# Global instance
_audit_writer = AsyncAuditWriter()

def start_async_audit_writer():
    """Called at system startup."""
    _audit_writer.start()

def stop_async_audit_writer(timeout_sec: int = 10):
    """Called at system shutdown."""
    _audit_writer.stop(timeout_sec)

def queue_audit_log(entry_type: str, entry):
    """Public API for audit logging."""
    _audit_writer.queue_log(entry_type, entry)
```

**Integration**:
```python
# In main.py at startup:
from audit.async_writer import start_async_audit_writer, stop_async_audit_writer

class TradingSystem:
    def __init__(self, ...):
        # ... existing init ...
        start_async_audit_writer()
    
    def __del__(self):
        stop_async_audit_writer()
```

**Testing**:
- Verify audit logs are written without blocking
- Measure order execution latency before/after

---

### Architectural Change 3: Trading Cycle Serialization (1-2 days)
**Fixes**: Issue #3, #6

**Scope**:
1. Add mutual exclusion lock to serialize equity + crypto cycles
2. Or: separate thread pools + independent capital reservation
3. Add cycle execution metrics

**Code**:
```python
# orchestrator/main.py
from threading import Lock

class TradingSystem:
    def __init__(self, market_type: str | None = None) -> None:
        # ... existing init ...
        self._trading_cycle_lock = Lock()  # Serialize equity + crypto
        self._cycle_start_time: float | None = None
    
    def trading_loop(self) -> None:
        """One scan-signal-execute cycle for the active universe(s)."""
        import time
        
        try:
            # ... existing setup (kill switch, circuit breaker, capital refresh) ...
            
            with self._trading_cycle_lock:
                self._cycle_start_time = time.time()
                
                # Equity cycle
                if self._market_type in ("equity", "both"):
                    self._update_regime()
                    if not self._regime_detector.should_suppress_new_entries(...):
                        self._run_equity_cycle()
                
                # Crypto cycle (runs after equity if both enabled)
                if self._market_type in ("crypto", "both"):
                    self._run_crypto_cycle()
                
                # Record cycle duration
                elapsed = time.time() - self._cycle_start_time
                log.info("trading_cycle_complete", duration_sec=elapsed)
        
        finally:
            HealthMonitor().write_heartbeat()
```

**Complexity**: 1/5 (add lock + timing)

**Testing**:
- Verify no concurrent order placement
- Monitor cycle duration over time

---

## Summary Table

| Issue | Severity | Type | Quick Fix? | Effort | Impact |
|-------|----------|------|-----------|--------|--------|
| #1: APScheduler not HA | CRITICAL | Config | No | 4 hrs | Jobs lost on crash |
| #2: Silent job failures | CRITICAL | Observability | Yes (Win #3) | 45 min | Missed trading cycles |
| #3: Overlapping cycles race | HIGH | Concurrency | Yes (Win #3 scope) | 1-2 hrs | Capital overallocation |
| #4: State accumulation | HIGH | Memory leak | Yes (Win #1) | 1 hr | OOM crash, duplicates |
| #5: Blocking audit logs | HIGH | Performance | Partial | 3 days | Order latency |
| #6: Post-market timing | MEDIUM | Ordering | Yes | 30 min | Incomplete summaries |
| #7: A/B routing race | MEDIUM | Race condition | Yes (Win #2) | 30 min | Invalid test results |
| #8: Sync DB writes | MEDIUM | Performance | Partial | 2 days | Trade latency |
| #9: Runner state leak | MEDIUM | Memory leak | Yes (Win #1) | 10 min | Duplicate trades |
| #10: Report conflicts | MEDIUM | Scheduling | Yes | 30 min | Hung reports |
| #11: Job persistence | HIGH | Architecture | No | 3-4 days | No restart recovery |
| #12: No report timeout | MEDIUM | Reliability | No | 2 hrs | Hung scheduler |
| #13: No graceful shutdown | MEDIUM | Reliability | No | 1-2 hrs | Hung jobs |
| #14: Regime not sync'd | LOW | Logic | No | 1 hr | Unhedged positions |

---

## Recommended Priority Order

### Immediate (This Week)
1. ✅ **Quick Win #1**: Clear state (30 min)
2. ✅ **Quick Win #2**: Fix A/B routing (30 min)
3. ✅ **Quick Win #3**: Enhance _safe wrapper (45 min)
4. 🔧 **Issue #6**: Post-market timing (30 min)
5. 🔧 **Issue #10**: Report schedule conflict (30 min)

### Short-term (Next 2 Weeks)
6. 🔧 **Architectural Change #3**: Cycle serialization (1-2 days)
7. 🔧 **Architectural Change #1**: APScheduler HA (3-4 days, requires testing)

### Medium-term (Next Month)
8. 🔧 **Architectural Change #2**: Async audit logging (2-3 days)
9. 🔧 **Issue #12**: Report timeout circuit breaker (2 hrs)
10. 🔧 **Issue #13**: Graceful shutdown (1-2 hrs)

---

## Validation Checklist

After implementing fixes:

- [ ] Memory profiling: No growth over 24 hours
- [ ] Job execution tracking: All scheduled jobs complete within timeout
- [ ] A/B test routing: 100 symbols show ~50/50 split over 7 days
- [ ] No duplicate trades: Same signal never processed twice
- [ ] Post-market runs: Always completes 5 min after last trading_loop
- [ ] Graceful restart: Lost jobs recovered from DB (if implemented)
- [ ] Telegram alerts: All job failures trigger alerts
- [ ] Load test: 100+ concurrent signals handled without race conditions

---

## Appendix: Configuration Template

```yaml
# config/scheduler.yaml (proposed)
scheduler:
  # Job store (Phase 7.5)
  job_store: "sqlalchemy"  # memory, sqlalchemy
  job_store_url: "postgresql://user:pass@localhost/trading_db"
  
  # Thread pool
  default_executor: "threadpool"
  default_executor_max_workers: 4
  blocking_executor_max_workers: 2
  
  # Job defaults
  coalesce: true  # Combine missed executions
  max_instances: 1  # Prevent concurrent runs
  misfire_grace_time_sec: 30  # Allow 30s late execution
  
  # Graceful shutdown
  shutdown_timeout_sec: 60
  
  # Audit logging
  async_audit: true
  audit_queue_size: 10000
  
  # Reporting
  report_job_timeout_sec: 300
  report_max_workers: 1

# Scheduler job timing overrides (IST)
jobs:
  equity_pre_market: "08:45"
  equity_trading_loop: "09:15-15:25/5"  # Every 5 min
  equity_post_market: "15:35"
  daily_reporting: "16:00"
  weekly_reporting: "fri 17:00"
  monthly_reporting: "eom 17:00"
```

---

## References

- APScheduler docs: https://apscheduler.readthedocs.io/en/latest/
- SQLAlchemy job store: https://apscheduler.readthedocs.io/en/latest/modules/jobstores/sqlalchemy.html
- Threading best practices: https://docs.python.org/3/library/threading.html
- Async patterns in Python: https://docs.python.org/3/library/asyncio.html

---

**Review Date**: 2025-01-XX  
**Reviewed By**: System Architect  
**Status**: Ready for implementation  
