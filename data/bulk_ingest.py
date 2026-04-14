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

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import structlog
from sqlalchemy import text

from config.settings import settings
from data.clean import prepare_ohlcv
from data.ingest import KiteIngestor
from data.rate_limiter import RateLimiter
from data.store import _OHLCV_COLS, get_engine, write_ohlcv_records

log = structlog.get_logger(__name__)

_KITE_RPS = settings.kite_rps
_BATCH_SIZE = settings.bulk_ingest_batch_size
_MAX_WORKERS = settings.bulk_ingest_max_workers
_DB_WORKERS = settings.bulk_ingest_db_workers


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


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
    limiter: RateLimiter,
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
    token = instrument["instrument_token"]
    symbol = instrument["tradingsymbol"]

    # Skip if already up to date
    latest = _latest_date_in_db(token, interval)
    if latest and latest >= to_date:
        log.debug("skip_already_current", symbol=symbol, latest=latest)
        return symbol, 0, []

    # Advance from_date if partial data exists
    effective_from = latest + timedelta(days=1) if latest else from_date

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
    df["token"] = token
    df["symbol"] = symbol
    df["interval"] = interval

    df = prepare_ohlcv(df, interval=interval)
    if df is None:
        log.warning("invalid_ohlcv_skipped", symbol=symbol)
        return symbol, 0, []

    df = df.reset_index()
    if df.columns[0] != "time":
        df = df.rename(columns={df.columns[0]: "time"})

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
    limiter = RateLimiter(_KITE_RPS)
    results: dict[str, int] = {}
    pending_rows: list[dict] = []
    total_written = 0
    total = len(instruments)
    flush_futures: list[Future] = []

    log.info(
        "bulk_ingest_start",
        symbols=total,
        from_date=str(from_date),
        to_date=str(to_date),
        interval=interval,
        workers=_MAX_WORKERS,
        db_workers=_DB_WORKERS,
    )

    # Two thread pools: one for API fetches, one for DB writes.
    # DB writes overlap with ongoing API fetches instead of blocking them.
    with (
        ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="kite") as fetch_pool,
        ThreadPoolExecutor(max_workers=_DB_WORKERS, thread_name_prefix="db") as db_pool,
    ):
        fetch_futures = {
            fetch_pool.submit(
                _fetch_one, ingestor, limiter, inst, from_date, to_date, interval
            ): inst["tradingsymbol"]
            for inst in instruments
        }

        for done, future in enumerate(as_completed(fetch_futures), 1):
            symbol, count, rows = future.result()
            results[symbol] = count
            pending_rows.extend(rows)

            # Dispatch a non-blocking flush when batch is full
            if len(pending_rows) >= _BATCH_SIZE:
                batch = pending_rows[:]
                pending_rows.clear()
                flush_futures.append(db_pool.submit(write_ohlcv_records, batch))
                log.info(
                    "bulk_ingest_progress",
                    done=done,
                    total=total,
                    queued_rows=len(batch),
                )

        # Flush remainder (still non-blocking — wait below)
        if pending_rows:
            flush_futures.append(db_pool.submit(write_ohlcv_records, pending_rows[:]))

        # Wait for all DB writes and accumulate written counts
        for ff in flush_futures:
            total_written += ff.result()

    log.info(
        "bulk_ingest_complete",
        symbols_processed=total,
        total_rows_written=total_written,
    )
    return results
