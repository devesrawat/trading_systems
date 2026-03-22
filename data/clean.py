"""
Data quality helpers for OHLCV bars.

Covers:
  - Outlier removal (z-score or IQR)
  - Corporate action back-adjustment (splits via Kite)
  - Missing bar forward-fill (max 3 consecutive)
  - OHLCV sanity validation
  - NSE circuit-limit day flagging
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from kiteconnect import KiteConnect

log = structlog.get_logger(__name__)

NSE_CIRCUIT_THRESHOLD = 0.199   # 19.9% — NSE individual stock circuit limit


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers(
    df: pd.DataFrame,
    col: str = "close",
    method: str = "zscore",
    threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Remove rows where *col* is an outlier.

    method='zscore'  — drop rows where |z-score| > threshold
    method='iqr'     — drop rows outside (Q1 - k*IQR, Q3 + k*IQR), k=threshold
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    series = df[col].astype(float)

    if method == "zscore":
        mean = series.mean()
        std = series.std()
        if std == 0:
            return df
        mask = ((series - mean) / std).abs() <= threshold
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - threshold * iqr
        hi = q3 + threshold * iqr
        mask = series.between(lo, hi)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'zscore' or 'iqr'.")

    removed = (~mask).sum()
    if removed > 0:
        log.warning("outliers_removed", col=col, method=method, count=int(removed))
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Corporate action adjustments
# ---------------------------------------------------------------------------

def adjust_splits(df: pd.DataFrame, token: int, kite: "KiteConnect") -> pd.DataFrame:
    """
    Back-adjust OHLCV for stock splits using Kite corporate action data.

    Applies a multiplicative adjustment factor to close, open, high, low
    and a divisive factor to volume for each split event prior to ex-date.
    Returns a new DataFrame — never mutates the input.
    """
    try:
        corporate_actions = kite.get_corporate_actions(token)
    except Exception as exc:
        log.error("corporate_action_fetch_failed", token=token, error=str(exc))
        return df

    splits = [
        ca for ca in (corporate_actions or [])
        if ca.get("type", "").lower() in ("split", "bonus")
    ]
    if not splits:
        return df

    result = df.copy()
    result = result.sort_index()

    for split in sorted(splits, key=lambda x: x["ex_date"], reverse=True):
        ex_date = pd.Timestamp(split["ex_date"])
        ratio: float = float(split.get("ratio", 1.0))   # e.g. 2.0 for 2-for-1 split

        if ratio <= 0 or ratio == 1.0:
            continue

        pre_mask = result.index < ex_date
        price_cols = [c for c in ["open", "high", "low", "close"] if c in result.columns]
        result.loc[pre_mask, price_cols] = result.loc[pre_mask, price_cols] / ratio
        if "volume" in result.columns:
            result.loc[pre_mask, "volume"] = (result.loc[pre_mask, "volume"] * ratio).astype("int64")

        log.info("split_adjusted", token=token, ex_date=ex_date, ratio=ratio)

    return result


# ---------------------------------------------------------------------------
# Missing bar handling
# ---------------------------------------------------------------------------

def fill_missing_bars(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Forward-fill gaps up to 3 consecutive missing bars.
    Rows representing gaps > 3 bars are marked with 'is_filled=True'.

    For 'day' interval, reindexes on business days.
    For intraday intervals, reindexes on the expected intraday grid.
    """
    if df.empty:
        return df

    result = df.copy()
    result["is_filled"] = False

    freq_map = {
        "minute": "1min",
        "3minute": "3min",
        "5minute": "5min",
        "15minute": "15min",
        "30minute": "30min",
        "60minute": "60min",
        "day": "B",
    }
    freq = freq_map.get(interval)
    if freq is None:
        log.warning("unknown_interval_for_fill", interval=interval)
        return result

    full_index = pd.date_range(result.index.min(), result.index.max(), freq=freq)
    result = result.reindex(full_index)

    # Mark newly introduced rows as filled
    newly_filled = result["is_filled"].isna()

    # Forward fill, but track consecutive run length
    consec_fill = newly_filled.astype(int)
    # Running count of consecutive NaN blocks
    block_id = (~newly_filled).cumsum()
    run_lengths = consec_fill.groupby(block_id).cumsum()

    # Only fill where run length ≤ 3
    fillable = run_lengths <= 3
    result = result.ffill()
    result.loc[~fillable & newly_filled, :] = np.nan  # leave long gaps as NaN

    result["is_filled"] = newly_filled & fillable
    result.index.name = "time"

    filled_count = int(newly_filled.sum())
    if filled_count:
        log.info("bars_filled", interval=interval, filled=filled_count)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Sanity-check an OHLCV DataFrame.

    Returns (is_valid, list_of_issues).
    """
    issues: list[str] = []

    required = {"open", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return False, issues

    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues

    # high >= low
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        issues.append(f"high < low on {int(bad_hl.sum())} rows")

    # close within [low, high]
    bad_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if bad_close.any():
        issues.append(f"close outside [low, high] on {int(bad_close.sum())} rows")

    # open within [low, high]
    bad_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if bad_open.any():
        issues.append(f"open outside [low, high] on {int(bad_open.sum())} rows")

    # non-negative volume
    if "volume" in df.columns:
        bad_vol = df["volume"] < 0
        if bad_vol.any():
            issues.append(f"Negative volume on {int(bad_vol.sum())} rows")

    is_valid = len(issues) == 0
    if not is_valid:
        log.warning("ohlcv_validation_failed", issues=issues)
    return is_valid, issues


# ---------------------------------------------------------------------------
# Canonical preparation pipeline  (single source of truth)
# ---------------------------------------------------------------------------

def prepare_ohlcv(
    df: pd.DataFrame,
    interval: str = "day",
    min_bars: int = 0,
) -> pd.DataFrame | None:
    """
    Standard OHLCV preparation pipeline used across all strategies and ingestion.

    Steps
    -----
    1. Validate OHLCV columns and basic sanity (high ≥ low, etc.)
    2. Remove close and volume outliers (z-score, threshold=4)
    3. For daily bars: forward-fill gaps ≤ 3, flag and drop circuit-limit days
    4. Drop helper columns (``is_filled``, ``circuit_hit``)
    5. Enforce *min_bars* floor — return None if too few bars remain

    Parameters
    ----------
    df        : raw OHLCV DataFrame; may have a ``time`` column **or** a
                ``DatetimeIndex`` named ``"time"``.
    interval  : Kite/canonical interval string (``"day"``, ``"5minute"``, …).
    min_bars  : return None if ``len(result) < min_bars``.  0 = no floor.

    Returns
    -------
    Clean DataFrame with a ``DatetimeIndex`` named ``"time"``, or ``None``
    when data quality is insufficient.
    """
    is_valid, _ = validate_ohlcv(df)
    if not is_valid:
        return None

    if "time" in df.columns and df.index.name != "time":
        df = df.set_index("time")

    df = remove_outliers(df, col="close", method="zscore", threshold=4.0)
    df = remove_outliers(df, col="volume", method="zscore", threshold=4.0)

    if interval == "day":
        df = fill_missing_bars(df, interval="day")
        df = flag_circuit_limit_days(df)
        df = df[~df["circuit_hit"]].drop(columns=["circuit_hit"])

    if "is_filled" in df.columns:
        df = df.drop(columns=["is_filled"])

    if min_bars and len(df) < min_bars:
        return None

    return df


# ---------------------------------------------------------------------------
# NSE circuit limit flagging
# ---------------------------------------------------------------------------

def flag_circuit_limit_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'circuit_hit' column.

    A circuit is considered hit when the absolute daily percentage change
    in close price exceeds 19.9% (NSE individual stock circuit filter).
    Only meaningful on daily ('day') interval bars.
    """
    result = df.copy()
    pct_chg = result["close"].pct_change().abs()
    result["circuit_hit"] = pct_chg > NSE_CIRCUIT_THRESHOLD
    circuit_days = int(result["circuit_hit"].sum())
    if circuit_days:
        log.info("circuit_days_flagged", count=circuit_days)
    return result
