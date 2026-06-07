"""
Correlation analysis for portfolio positions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog

from data.store import get_engine, get_redis
from portfolio.schema import PortfolioPosition

log = structlog.get_logger(__name__)


def compute_pairwise_correlation(symbol1: str, symbol2: str, lookback_days: int = 252) -> float:
    """
    Compute pairwise correlation between two symbols using historical data from TimescaleDB.
    Results are cached in Redis for 24 hours.

    Parameters
    ----------
    symbol1 : str
        First symbol.
    symbol2 : str
        Second symbol.
    lookback_days : int
        Lookback window in days.

    Returns
    -------
    float
        Correlation coefficient [-1, 1]. Returns 0.0 if data is insufficient.
    """
    # 1. Check Redis cache first
    r = get_redis()
    # Sort symbols to ensure consistent cache keys regardless of order
    s1, s2 = sorted([symbol1, symbol2])
    cache_key = f"portfolio:corr:{s1}:{s2}:{lookback_days}"

    cached = r.get(cache_key)
    if cached is not None:
        return float(cached)

    # 2. Fetch data from TimescaleDB
    try:
        engine = get_engine()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer for holidays

        query = """
            SELECT time, symbol, close
            FROM ohlcv
            WHERE symbol IN (:s1, :s2)
              AND interval = 'day'
              AND time >= :start_date
              AND time <= :end_date
            ORDER BY time ASC
        """

        with engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"s1": s1, "s2": s2, "start_date": start_date, "end_date": end_date},
                parse_dates=["time"],
            )

        if df.empty or len(df["symbol"].unique()) < 2:
            log.warning("correlation_insufficient_data", symbol1=s1, symbol2=s2)
            return 0.0

        # Pivot to get time as index and symbols as columns
        pivot_df = df.pivot(index="time", columns="symbol", values="close").dropna()

        if len(pivot_df) < min(lookback_days // 2, 20):
            log.warning(
                "correlation_insufficient_overlap", symbol1=s1, symbol2=s2, rows=len(pivot_df)
            )
            return 0.0

        # 3. Compute returns and correlation
        returns = pivot_df.pct_change().dropna()
        correlation = float(returns[s1].corr(returns[s2]))

        if np.isnan(correlation):
            correlation = 0.0

        # 4. Cache and return
        r.setex(cache_key, 86400, str(correlation))  # 24h TTL
        log.debug("correlation_computed", symbol1=s1, symbol2=s2, corr=round(correlation, 3))
        return correlation

    except Exception as exc:
        log.error("correlation_compute_failed", symbol1=s1, symbol2=s2, error=str(exc))
        return 0.0


def compute_portfolio_correlation(
    symbol: str,
    current_positions: dict[str, PortfolioPosition],
    lookback_days: int = 252,
) -> tuple[float, tuple[str, str] | None]:
    """
    Compute average correlation of a symbol to current portfolio.

    Parameters
    ----------
    symbol : str
        Symbol being evaluated.
    current_positions : dict[str, object]
        Current open positions (key: symbol).
    lookback_days : int
        Lookback window in days.

    Returns
    -------
    tuple[float, tuple[str, str] | None]
        (average_correlation, (symbol1, symbol2) with highest correlation)
        Returns (0.0, None) if no open positions.
    """
    if not current_positions:
        return 0.0, None

    correlations = []
    max_corr = -2.0
    max_pair: tuple[str, str] | None = None

    for existing_symbol in current_positions:
        corr = compute_pairwise_correlation(symbol, existing_symbol, lookback_days)
        correlations.append(corr)
        if corr > max_corr:
            max_corr = corr
            max_pair = (symbol, existing_symbol)

    avg_corr = sum(correlations) / len(correlations) if correlations else 0.0
    return avg_corr, max_pair if max_corr > -2.0 else None


def apply_correlation_penalty(confidence: float, avg_correlation: float) -> float:
    """
    Apply a penalty to signal confidence based on portfolio correlation.

    Logic:
    - If avg_correlation < 0.3: no penalty
    - If avg_correlation 0.3-0.6: linear penalty (0-0.5x confidence)
    - If avg_correlation > 0.6: strong penalty (0.5x confidence)

    Parameters
    ----------
    confidence : float
        Original signal confidence [0, 1].
    avg_correlation : float
        Average correlation to portfolio [-1, 1].

    Returns
    -------
    float
        Adjusted confidence [0, 1].
    """
    if avg_correlation < 0.3:
        penalty = 0.0
    elif avg_correlation < 0.6:
        # Linear from 0 to 0.5 as correlation goes from 0.3 to 0.6
        penalty = ((avg_correlation - 0.3) / 0.3) * 0.5
    else:
        # Strong penalty
        penalty = 0.5

    adjusted = confidence * (1.0 - penalty)
    return adjusted


def is_redundant_position(
    symbol: str,
    current_positions: dict[str, PortfolioPosition],
    threshold: float = 0.6,
) -> bool:
    """
    Check if a symbol would be redundant (highly correlated to existing positions).

    Parameters
    ----------
    symbol : str
        Symbol being evaluated.
    current_positions : dict[str, object]
        Current open positions.
    threshold : float
        Correlation threshold above which position is considered redundant.

    Returns
    -------
    bool
        True if symbol is redundant.
    """
    if not current_positions:
        return False

    avg_corr, _ = compute_portfolio_correlation(symbol, current_positions)
    return avg_corr > threshold
