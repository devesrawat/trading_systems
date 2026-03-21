"""
IV-based feature engineering for NSE F&O options strategies.

Public API
----------
build_fo_features(symbol, expiry_date, kite) → IVFeatures
compute_iv_rank(iv_series)   → float [0–1]
compute_iv_percentile(iv_series) → float [0–100]
compute_realized_vol(price_series, window) → float (annualised)
compute_max_pain(strikes, call_oi, put_oi) → int
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class IVFeatures:
    """Snapshot of IV-derived features for one symbol / expiry."""

    symbol: str
    expiry_date: date
    iv_rank: float          # 0–1: position of current IV in 52-week range
    iv_percentile: float    # 0–100: percentage of days current IV exceeded
    iv_premium: float       # implied_vol - realized_vol (positive = IV elevated)
    put_call_ratio: float   # total put OI / total call OI
    max_pain: int           # strike at which option writers profit most
    days_to_expiry: int
    current_iv: float       # most recent implied volatility
    realized_vol: float     # 20-day annualised realized vol of underlying

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core feature calculators
# ---------------------------------------------------------------------------


def compute_iv_rank(iv_series: pd.Series) -> float:
    """
    IV Rank = (current_IV - 52w_low) / (52w_high - 52w_low).

    Uses the *last* element of iv_series as the current value.
    Returns 0.0 when the range is flat (high == low).
    """
    if iv_series.empty:
        return 0.0
    current = float(iv_series.iloc[-1])
    low = float(iv_series.min())
    high = float(iv_series.max())
    if high == low:
        return 0.0
    return float(np.clip((current - low) / (high - low), 0.0, 1.0))


def compute_iv_percentile(iv_series: pd.Series) -> float:
    """
    IV Percentile = fraction of historical days on which current IV exceeded
    those past values × 100.

    Uses the *last* element as current; the rest are the historical window.
    Returns 0.0 for empty or single-element series.
    """
    if len(iv_series) < 2:
        return 0.0
    current = float(iv_series.iloc[-1])
    history = iv_series.iloc[:-1]
    pct = float((history < current).mean() * 100.0)
    return float(np.clip(pct, 0.0, 100.0))


def compute_realized_vol(price_series: pd.Series, window: int = 20) -> float:
    """
    Annualised realised volatility from the last *window* log-returns.

    Returns 0.0 if the series is too short or perfectly flat.
    """
    if len(price_series) < 2:
        return 0.0
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_returns) == 0:
        return 0.0
    tail = log_returns.iloc[-window:] if len(log_returns) >= window else log_returns
    rv = float(tail.std(ddof=1)) * np.sqrt(252)
    return max(rv, 0.0)


def compute_max_pain(
    strikes: list[int],
    call_oi: dict[int, int],
    put_oi: dict[int, int],
) -> int:
    """
    Max pain = strike at which total option writer profit is maximised.

    At a given expiry price P:
      - call writers profit from all calls with strike > P (those expire worthless)
        → worth 0; writers already collected premium, but we model pain as:
        call_pain(P) = Σ_{K > P} (K - P) * call_OI[K]
        put_pain(P)  = Σ_{K < P} (P - K) * put_OI[K]
    Max pain = argmin of total_pain(P).
    """
    best_strike = strikes[0]
    best_pain = float("inf")
    for s in strikes:
        call_pain = sum(
            max(k - s, 0) * oi for k, oi in call_oi.items()
        )
        put_pain = sum(
            max(s - k, 0) * oi for k, oi in put_oi.items()
        )
        total = call_pain + put_pain
        if total < best_pain:
            best_pain = total
            best_strike = s
    return best_strike


# ---------------------------------------------------------------------------
# Private helpers (patched in tests)
# ---------------------------------------------------------------------------


def _fetch_iv_history(symbol: str, kite: Any) -> pd.Series:
    """
    Fetch 252 trading days of implied volatility from Kite historical data.
    Falls back to realised vol as proxy when IV data is unavailable.
    """
    try:
        data = kite.historical_data(
            instrument_token=symbol,
            from_date=None,
            to_date=None,
            interval="day",
        )
        iv_values = [row.get("iv", row.get("close", 0)) for row in data]
        return pd.Series(iv_values, dtype=float)
    except Exception as exc:
        log.warning("iv_history_fetch_failed", symbol=symbol, error=str(exc))
        return pd.Series(dtype=float)


def _fetch_option_chain(
    symbol: str,
    expiry_date: date,
    kite: Any,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Return (call_oi, put_oi) dicts keyed by strike from Kite option chain.
    Returns empty dicts on failure.
    """
    try:
        instruments = kite.instruments("NFO")
        calls: dict[int, int] = {}
        puts: dict[int, int] = {}
        for inst in instruments:
            if inst.get("name") != symbol:
                continue
            exp = inst.get("expiry")
            if isinstance(exp, str):
                exp = datetime.strptime(exp, "%Y-%m-%d").date()
            if exp != expiry_date:
                continue
            strike = int(inst.get("strike", 0))
            oi = int(inst.get("oi", 0))
            if inst.get("instrument_type") == "CE":
                calls[strike] = calls.get(strike, 0) + oi
            elif inst.get("instrument_type") == "PE":
                puts[strike] = puts.get(strike, 0) + oi
        return calls, puts
    except Exception as exc:
        log.warning("option_chain_fetch_failed", symbol=symbol, error=str(exc))
        return {}, {}


def _fetch_underlying_prices(symbol: str, kite: Any) -> pd.Series:
    """Fetch last 30 days of daily close prices for the underlying."""
    try:
        data = kite.historical_data(
            instrument_token=symbol,
            from_date=None,
            to_date=None,
            interval="day",
        )
        closes = [row.get("close", 0) for row in data[-30:]]
        return pd.Series(closes, dtype=float)
    except Exception as exc:
        log.warning("underlying_prices_fetch_failed", symbol=symbol, error=str(exc))
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

_INSERT_IV_SNAPSHOT = text("""
    INSERT INTO iv_snapshots
        (time, symbol, expiry_date, iv_rank, iv_percentile, iv_premium,
         put_call_ratio, max_pain, days_to_expiry, current_iv, realized_vol)
    VALUES
        (:time, :symbol, :expiry_date, :iv_rank, :iv_percentile, :iv_premium,
         :put_call_ratio, :max_pain, :days_to_expiry, :current_iv, :realized_vol)
""")


def _write_iv_snapshot(features: IVFeatures, signal_type: str | None = None) -> None:
    """Persist an IVFeatures snapshot to the iv_snapshots hypertable."""
    from data.store import get_engine  # local import avoids circular dependency
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(_INSERT_IV_SNAPSHOT, {
                "time": datetime.now(tz=timezone.utc),
                "symbol": features.symbol,
                "expiry_date": features.expiry_date,
                "iv_rank": features.iv_rank,
                "iv_percentile": features.iv_percentile,
                "iv_premium": features.iv_premium,
                "put_call_ratio": features.put_call_ratio,
                "max_pain": features.max_pain,
                "days_to_expiry": features.days_to_expiry,
                "current_iv": features.current_iv,
                "realized_vol": features.realized_vol,
            })
            conn.commit()
        log.debug("iv_snapshot_written", symbol=features.symbol, expiry=str(features.expiry_date))
    except Exception as exc:
        log.warning("iv_snapshot_write_failed", symbol=features.symbol, error=str(exc))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_fo_features(symbol: str, expiry_date: date, kite: Any) -> IVFeatures:
    """
    Build all IV-based features for a given symbol / expiry.

    Parameters
    ----------
    symbol      : NSE symbol, e.g. "NIFTY" or "RELIANCE"
    expiry_date : the options contract expiry date
    kite        : authenticated KiteConnect instance (or mock)

    Returns
    -------
    IVFeatures dataclass with all fields populated.
    """
    iv_history = _fetch_iv_history(symbol, kite)
    call_oi, put_oi = _fetch_option_chain(symbol, expiry_date, kite)
    underlying_prices = _fetch_underlying_prices(symbol, kite)

    current_iv = float(iv_history.iloc[-1]) if not iv_history.empty else 0.0
    realized_vol = compute_realized_vol(underlying_prices, window=20)

    iv_rank = compute_iv_rank(iv_history)
    iv_percentile = compute_iv_percentile(iv_history)
    iv_premium = current_iv - realized_vol

    total_call_oi = sum(call_oi.values())
    total_put_oi = sum(put_oi.values())
    put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))
    max_pain_strike = compute_max_pain(strikes, call_oi, put_oi) if strikes else 0

    days_to_expiry = max(0, (expiry_date - date.today()).days)

    features = IVFeatures(
        symbol=symbol,
        expiry_date=expiry_date,
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        iv_premium=iv_premium,
        put_call_ratio=put_call_ratio,
        max_pain=max_pain_strike,
        days_to_expiry=days_to_expiry,
        current_iv=current_iv,
        realized_vol=realized_vol,
    )

    log.info(
        "fo_features_built",
        symbol=symbol,
        iv_rank=round(iv_rank, 3),
        iv_percentile=round(iv_percentile, 1),
        iv_premium=round(iv_premium, 4),
        put_call_ratio=round(put_call_ratio, 2),
        days_to_expiry=days_to_expiry,
    )

    _write_iv_snapshot(features)
    return features
