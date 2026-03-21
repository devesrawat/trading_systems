"""
Bulk historical ingest for 500+ instruments.

Design constraints:
  - Kite Connect rate limit: 3 historical-data requests / second
  - Daily bars need no chunking (single request per symbol)
  - Minute bars are already chunked by KiteIngestor._date_chunks()
  - DB writes are batched (BATCH_SIZE rows) to minimise round-trips
  - Symbols already up-to-date in DB are skipped (resume-safe)
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from threading import Semaphore
from typing import Any

import pandas as pd
import structlog

from data.clean import (
    fill_missing_bars,
    flag_circuit_limit_days,
    remove_outliers,
    validate_ohlcv,
)
from data.ingest import KiteIngestor
from data.store import get_engine, get_ohlcv
from sqlalchemy import text

log = structlog.get_logger(__name__)

# Kite allows 3 historical requests per second per user
_KITE_RPS = 3
_BATCH_SIZE = 500        # rows to accumulate before a DB flush
_MAX_WORKERS = 6         # threads — half wait on API, half wait on DB


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Leaky-bucket: allows at most *rps* calls per second."""

    def __init__(self, rps: int) -> None:
        self._sem = Semaphore(rps)
        self._rps = rps

    def __enter__(self) -> "_RateLimiter":
        self._sem.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        # Release the slot after 1 second so the rate stays at rps
        def _release(sem: Semaphore) -> None:
            time.sleep(1.0 / self._rps)
            sem.release()

        from threading import Thread
        Thread(target=_release, args=(self._sem,), daemon=True).start()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_UPSERT_SQL = text("""
    INSERT INTO ohlcv (time, token, symbol, open, high, low, close, volume, interval)
    VALUES (:time, :token, :symbol, :open, :high, :low, :close, :volume, :interval)
    ON CONFLICT DO NOTHING
""")

_OHLCV_COLS = ["time", "token", "symbol", "open", "high", "low", "close", "volume", "interval"]


def _flush(rows: list[dict]) -> int:
    """Write a batch of OHLCV rows to TimescaleDB. Returns row count written."""
    if not rows:
        return 0
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(_UPSERT_SQL, rows)
        conn.commit()
    return len(rows)


def _latest_date_in_db(token: int, interval: str) -> date | None:
    """Return the most recent bar date for this token+interval, or None."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MAX(time) FROM ohlcv WHERE token = :t AND interval = :i"),
            {"t": token, "i": interval},
        ).fetchone()
    return row[0].date() if row and row[0] else None


# ---------------------------------------------------------------------------
# Per-symbol fetch + clean
# ---------------------------------------------------------------------------

def _fetch_one(
    ingestor: KiteIngestor,
    limiter: _RateLimiter,
    instrument: dict,
    from_date: date,
    to_date: date,
    interval: str,
) -> tuple[str, int, list[dict]]:
    """
    Fetch, validate, and clean bars for a single symbol.

    Returns (symbol, row_count, rows_for_db).
    Skips symbols whose DB is already current to today.
    """
    token  = instrument["instrument_token"]
    symbol = instrument["tradingsymbol"]

    # Skip if already up to date
    latest = _latest_date_in_db(token, interval)
    if latest and latest >= to_date:
        log.debug("skip_already_current", symbol=symbol, latest=latest)
        return symbol, 0, []

    # Advance from_date if partial data exists
    effective_from = (
        latest + timedelta(days=1) if latest else from_date
    )

    with limiter:
        try:
            raw = ingestor.kite.historical_data(
                instrument_token=token,
                from_date=effective_from,
                to_date=to_date,
                interval=interval,
            )
        except Exception as exc:
            log.error("fetch_failed", symbol=symbol, error=str(exc))
            return symbol, 0, []

    if not raw:
        return symbol, 0, []

    df = pd.DataFrame(raw)
    df.rename(columns={"date": "time"}, inplace=True)
    df["token"]    = token
    df["symbol"]   = symbol
    df["interval"] = interval

    # --- clean pipeline ---
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    if not ohlcv_cols.issubset(df.columns):
        return symbol, 0, []

    is_valid, issues = validate_ohlcv(df)
    if not is_valid:
        log.warning("invalid_ohlcv_skipped", symbol=symbol, issues=issues)
        return symbol, 0, []

    df = df.set_index("time")
    df = remove_outliers(df, col="close", method="zscore", threshold=4.0)
    df = remove_outliers(df, col="volume", method="zscore", threshold=4.0)

    if interval == "day":
        df = fill_missing_bars(df.reset_index().rename(columns={"index": "time"}).set_index("time"), interval="day")
        df = flag_circuit_limit_days(df)
        df = df[~df["circuit_hit"]].drop(columns=["circuit_hit"])

    if "is_filled" in df.columns:
        df = df.drop(columns=["is_filled"])

    df = df.reset_index().rename(columns={"index": "time"})

    rows = df[_OHLCV_COLS].to_dict(orient="records")
    return symbol, len(rows), rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bulk_ingest(
    ingestor: KiteIngestor,
    instruments: list[dict],
    from_date: date,
    to_date: date,
    interval: str = "day",
) -> dict[str, int]:
    """
    Fetch and store historical OHLCV for up to 500+ instruments concurrently.

    Respects Kite's 3 req/s rate limit. Writes are batched (_BATCH_SIZE rows).
    Already-current symbols are skipped automatically (safe to re-run).

    Parameters
    ----------
    ingestor    : authenticated KiteIngestor
    instruments : list of instrument dicts from data.store.get_universe()
    from_date   : history start (inclusive)
    to_date     : history end (inclusive)
    interval    : Kite interval string, default 'day'

    Returns
    -------
    dict mapping symbol → rows written (0 = skipped or error)
    """
    limiter  = _RateLimiter(_KITE_RPS)
    results: dict[str, int] = {}
    pending_rows: list[dict] = []
    total_written = 0
    total = len(instruments)

    log.info(
        "bulk_ingest_start",
        symbols=total,
        from_date=str(from_date),
        to_date=str(to_date),
        interval=interval,
        workers=_MAX_WORKERS,
    )

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                _fetch_one, ingestor, limiter, inst, from_date, to_date, interval
            ): inst["tradingsymbol"]
            for inst in instruments
        }

        done = 0
        for future in as_completed(futures):
            symbol, count, rows = future.result()
            results[symbol] = count
            pending_rows.extend(rows)
            done += 1

            # Flush when batch is full
            if len(pending_rows) >= _BATCH_SIZE:
                written = _flush(pending_rows)
                total_written += written
                pending_rows.clear()
                log.info(
                    "bulk_ingest_progress",
                    done=done,
                    total=total,
                    flushed=written,
                    total_written=total_written,
                )

    # Flush remainder
    if pending_rows:
        written = _flush(pending_rows)
        total_written += written

    log.info(
        "bulk_ingest_complete",
        symbols_processed=total,
        total_rows_written=total_written,
    )
    return results
