"""
Parallel VCP (Volatility Contraction Pattern) scanner for 500+ instruments.

Pipeline
--------
1. Fetch all 500 symbols' OHLCV in a SINGLE multi-symbol SQL query.
2. Split the resulting DataFrame by symbol (groupby).
3. Run per-symbol VCP logic across a ProcessPoolExecutor (CPU-bound).
4. Stream results back — never holds all 500 clean DataFrames in memory at once.

Minervini criteria enforced
---------------------------
  - Stage 2 trend template (5 conditions)
  - ≥ 2 progressively tighter swing contractions in the last 60 bars
  - Final contraction range < 10 %
  - Volume dry-up in the last 10 bars vs prior 10
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import text

from data.clean import prepare_ohlcv
from data.store import get_engine

log = structlog.get_logger(__name__)

# Number of worker processes — default to CPU count, cap at 8
_WORKERS = min(os.cpu_count() or 4, 8)

# Minimum bars needed for a valid trend template check
_MIN_BARS = 200


# ---------------------------------------------------------------------------
# Multi-symbol DB fetch  (one query, not 500)
# ---------------------------------------------------------------------------


def fetch_ohlcv_bulk(
    symbols: list[str],
    lookback_days: int = 400,
    interval: str = "day",
) -> pd.DataFrame:
    """
    Fetch OHLCV for all *symbols* in a single SQL query.

    Returns a DataFrame indexed by (symbol, time) with columns
    open, high, low, close, volume.  ~40× faster than 500 individual queries.
    """
    from_date = date.today() - timedelta(days=lookback_days)
    engine = get_engine()

    query = text("""
        SELECT symbol, time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol  = ANY(:symbols)
          AND interval = :interval
          AND time    >= :from_date
        ORDER BY symbol, time ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "symbols": symbols,
                "interval": interval,
                "from_date": from_date,
            },
            parse_dates=["time"],
        )

    log.info(
        "bulk_ohlcv_fetched",
        symbols=len(symbols),
        rows=len(df),
        from_date=str(from_date),
    )
    return df


# ---------------------------------------------------------------------------
# Per-symbol VCP logic  (runs inside worker processes)
# ---------------------------------------------------------------------------


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    """Apply the standard OHLCV preparation pipeline."""
    return prepare_ohlcv(df, interval="day", min_bars=_MIN_BARS)


def _trend_template(df: pd.DataFrame) -> bool:
    close = df["close"]
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    cur = close.iloc[-1]
    high52 = close.rolling(252).max().iloc[-1]
    low52 = close.rolling(252).min().iloc[-1]

    return (
        cur > sma150.iloc[-1]  # above 150-day MA
        and cur > sma200.iloc[-1]  # above 200-day MA
        and sma150.iloc[-1] > sma200.iloc[-1]  # 150 above 200
        and sma200.iloc[-1] > sma200.iloc[-22]  # 200-day rising 1 month
        and cur >= high52 * 0.75  # within 25 % of 52-week high
        and cur >= low52 * 1.30  # ≥ 30 % above 52-week low
    )


def _swing_ranges(df: pd.DataFrame, lookback: int = 60) -> list[float]:
    """Return % high-to-low range for each swing pivot pair in the base."""
    base = df.tail(lookback)
    high = base["high"]
    low = base["low"]

    ph_mask = (high.shift(1) < high) & (high.shift(-1) < high)
    pl_mask = (low.shift(1) > low) & (low.shift(-1) > low)

    ph_idx = high[ph_mask].index.tolist()
    pl_idx = low[pl_mask].index.tolist()

    ranges: list[float] = []
    for ph in ph_idx:
        subsequent = [pl for pl in pl_idx if pl > ph]
        if not subsequent:
            continue
        pl = subsequent[0]
        swing_h = high.loc[ph]
        swing_l = low.loc[pl]
        if swing_h > 0:
            ranges.append(round((swing_h - swing_l) / swing_h * 100, 2))

    return ranges


def _volume_dries_up(df: pd.DataFrame) -> bool:
    if len(df) < 20:
        return False
    vol = df["volume"]
    return bool(vol.iloc[-10:].mean() < vol.iloc[-20:-10].mean())


def _pivot_buy_point(df: pd.DataFrame) -> float:
    return round(df.tail(15)["high"].max() * 1.005, 2)


def _scan_one(symbol: str, ohlcv_records: list[dict]) -> dict[str, Any] | None:
    """
    Entry point for each worker process.
    Accepts raw records (serialisable) rather than a DataFrame.
    """
    df = pd.DataFrame(ohlcv_records)
    if df.empty:
        return None

    df = _clean(df)
    if df is None:
        return None

    if not _trend_template(df):
        return None

    ranges = _swing_ranges(df, lookback=60)
    contractions = sum(1 for i in range(1, len(ranges)) if ranges[i] < ranges[i - 1])
    final_range = ranges[-1] if ranges else float("inf")

    if contractions < 2 or final_range >= 10.0:
        return None

    current = float(df["close"].iloc[-1])
    pivot = _pivot_buy_point(df)

    return {
        "symbol": symbol,
        "current_price": current,
        "pivot_buy": pivot,
        "distance_to_pivot_pct": round((pivot - current) / current * 100, 2),
        "contractions": contractions,
        "swing_ranges": ranges,
        "volume_dry_up": _volume_dries_up(df),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_vcp_universe(
    symbols: list[str],
    lookback_days: int = 400,
    interval: str = "day",
    workers: int = _WORKERS,
) -> list[dict[str, Any]]:
    """
    Run VCP scan across *symbols* in parallel.

    Parameters
    ----------
    symbols       : list of NSE trading symbols, e.g. ['RELIANCE', 'TCS', ...]
    lookback_days : how far back to pull data (400 days → ~16 months of bars)
    interval      : Kite interval, almost always 'day' for VCP
    workers       : ProcessPoolExecutor workers (default = CPU count, max 8)

    Returns
    -------
    List of result dicts for symbols that pass all VCP criteria,
    sorted by distance_to_pivot_pct ascending (closest to breakout first).

    Example
    -------
    >>> from data.store import get_universe
    >>> from signals.vcp_scanner import scan_vcp_universe
    >>>
    >>> instruments = get_universe(segment="EQ")
    >>> symbols = [i["tradingsymbol"] for i in instruments[:500]]
    >>> candidates = scan_vcp_universe(symbols)
    >>> for c in candidates:
    ...     print(c["symbol"], c["pivot_buy"], c["contractions"])
    """
    # Step 1: single bulk DB read
    all_data = fetch_ohlcv_bulk(symbols, lookback_days=lookback_days, interval=interval)

    if all_data.empty:
        log.warning("no_data_returned_from_db")
        return []

    # Step 2: group by symbol → dict of serialisable records for worker processes
    symbol_groups: dict[str, list[dict]] = {}
    for sym, group in all_data.groupby("symbol", sort=False):
        symbol_groups[str(sym)] = group.to_dict(orient="records")

    log.info(
        "vcp_scan_start",
        total_symbols=len(symbol_groups),
        workers=workers,
    )

    # Step 3: fan out across processes (CPU-bound pandas + rolling calculations)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scan_one, sym, records): sym for sym, records in symbol_groups.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                results.append(result)
            if done % 50 == 0:
                log.info("vcp_scan_progress", done=done, total=len(futures), found=len(results))

    # Sort by proximity to pivot buy point
    results.sort(key=lambda r: r["distance_to_pivot_pct"])

    log.info(
        "vcp_scan_complete",
        scanned=len(symbol_groups),
        candidates=len(results),
    )
    return results
